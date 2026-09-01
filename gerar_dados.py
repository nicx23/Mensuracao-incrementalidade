"""
Gera uma base sintética de campanhas de cartão com DUAS propriedades essenciais:

1. VIÉS DE SELEÇÃO: a campanha não foi aleatorizada. Quem gastava mais
   antes tinha mais chance de ser comunicado. É o caso real de quase toda
   operação de CRM — e é o que faz a comparação ingênua mentir.

2. EFEITO VERDADEIRO CONHECIDO: o contrafactual de cada cliente é gerado
   explicitamente, então sabemos o incremental real. Isso permite auditar
   se o método recupera a resposta certa.

Dado sintético, nenhuma informação de qualquer empresa.
"""

import os

import numpy as np
import pandas as pd

SEED = 42
N = 80_000

# incremental verdadeiro, em R$ por cliente comunicado, por estratégia
EFEITO_REAL = {
    "onboarding": 145.0,
    "iniciacao": 92.0,
    "rentabilizacao": 61.0,
    "reativacao": 38.0,
}


def gerar(n=N, seed=SEED):
    rng = np.random.default_rng(seed)

    seg = rng.choice(["A", "B", "C", "D"], size=n, p=[0.18, 0.32, 0.30, 0.20])
    nivel = pd.Series(seg).map({"A": 1.55, "B": 1.15, "C": 0.85, "D": 0.55}).values

    limite = np.exp(rng.normal(np.log(4200 * nivel), 0.55))
    meses_cliente = rng.integers(3, 120, size=n)
    freq_pre = rng.poisson(np.clip(9 * nivel, 1, None))
    gasto_pre = np.clip(
        rng.gamma(shape=2.4, scale=380 * nivel) + 0.045 * limite, 0, None
    )

    # --- atribuição NÃO aleatória: o time mirou quem já gastava mais ---
    z = (
        -1.35
        + 0.00042 * gasto_pre
        + 0.000075 * limite
        + 0.020 * freq_pre
        + rng.normal(0, 0.45, size=n)
    )
    p_comunicar = 1 / (1 + np.exp(-z))
    comunicado = rng.binomial(1, p_comunicar).astype(int)

    estrategia = np.where(
        comunicado == 1,
        rng.choice(list(EFEITO_REAL), size=n, p=[0.16, 0.24, 0.36, 0.24]),
        "nao_comunicado",
    )

    # --- contrafactual: o que teria acontecido sem nenhuma comunicação ---
    gasto_pos_sem = np.clip(
        0.83 * gasto_pre
        + 0.028 * limite
        + 24 * freq_pre
        + rng.normal(0, 260, size=n),
        0,
        None,
    )

    efeito = np.where(
        comunicado == 1,
        pd.Series(estrategia).map(EFEITO_REAL).fillna(0).values,
        0.0,
    )
    # heterogeneidade: o efeito é maior em quem tem mais folga de limite
    folga = np.clip(1 - gasto_pre / np.maximum(limite, 1), 0, 1)
    efeito = efeito * (0.55 + 0.9 * folga)

    gasto_pos = np.clip(gasto_pos_sem + efeito, 0, None)

    return pd.DataFrame(
        {
            "id_cliente": np.arange(1, n + 1),
            "segmento": seg,
            "limite": limite.round(2),
            "meses_cliente": meses_cliente,
            "freq_pre": freq_pre,
            "gasto_pre": gasto_pre.round(2),
            "comunicado": comunicado,
            "estrategia": estrategia,
            "gasto_pos": gasto_pos.round(2),
            # colunas de auditoria — não usadas pelo estimador
            "_gasto_pos_contrafactual": gasto_pos_sem.round(2),
            "_efeito_real": efeito.round(2),
        }
    )


if __name__ == "__main__":
    os.makedirs("dados", exist_ok=True)
    df = gerar()
    df.to_csv("dados/base_campanha.csv", index=False)
    att = df.loc[df.comunicado == 1, "_efeito_real"].mean()
    print(f"{len(df):,} clientes | comunicados: {df.comunicado.mean():.1%}")
    print(f"ATT verdadeiro: R$ {att:,.2f} por cliente comunicado")
