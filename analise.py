"""
Mensuração de faturamento incremental de campanha não aleatorizada.

Pipeline:
  1. Diagnóstico do viés de seleção
  2. Grupo de controle pareado 1:1 por KNN dentro de segmento, com caliper
  3. Checagem de balanceamento (SMD antes x depois)
  4. Estimativa do incremental + intervalo por bootstrap
  5. Teste de significância não paramétrico (Mann-Whitney)
  6. Quebra por estratégia
  7. Auditoria contra o efeito verdadeiro
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from sklearn.neighbors import NearestNeighbors

COVARIAVEIS = ["gasto_pre", "limite", "freq_pre", "meses_cliente"]

TINTA, FUNDO = "#0F2E34", "#FFFFFF"
AMBAR, TEAL, CINZA = "#D99B1C", "#3E7C82", "#9AA9AB"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Poppins", "DejaVu Sans"],
    "axes.edgecolor": "#D6DEDF", "axes.labelcolor": TINTA,
    "text.color": TINTA, "xtick.color": TINTA, "ytick.color": TINTA,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": FUNDO, "axes.facecolor": FUNDO, "figure.dpi": 140,
})


# ----------------------------------------------------------------- balanço
def smd(tratado, controle):
    """Diferença padronizada de médias. |SMD| < 0.10 = grupos comparáveis."""
    s = np.sqrt((tratado.var(ddof=1) + controle.var(ddof=1)) / 2)
    return (tratado.mean() - controle.mean()) / s if s > 0 else 0.0


def tabela_balanco(tratados, controles):
    return pd.DataFrame(
        {"covariavel": COVARIAVEIS,
         "media_tratado": [tratados[c].mean() for c in COVARIAVEIS],
         "media_controle": [controles[c].mean() for c in COVARIAVEIS],
         "smd": [smd(tratados[c], controles[c]) for c in COVARIAVEIS]}
    )


# --------------------------------------------------------------- matching
def parear(df, caliper=0.25, seed=42):
    """
    KNN 1:1 sem reposição, dentro de cada segmento comportamental.
    Covariáveis padronizadas; descarta pares acima do caliper (em desvios).
    """
    rng = np.random.default_rng(seed)
    pares = []

    for seg, bloco in df.groupby("segmento", sort=False):
        t = bloco[bloco.comunicado == 1]
        c = bloco[bloco.comunicado == 0]
        if len(t) == 0 or len(c) == 0:
            continue

        mu, sd = bloco[COVARIAVEIS].mean(), bloco[COVARIAVEIS].std(ddof=0)
        Xt = ((t[COVARIAVEIS] - mu) / sd).to_numpy()
        Xc = ((c[COVARIAVEIS] - mu) / sd).to_numpy()

        k = min(12, len(c))
        nn = NearestNeighbors(n_neighbors=k).fit(Xc)
        dist, idx = nn.kneighbors(Xt)

        # ordem aleatória evita que os primeiros tratados fiquem com os
        # melhores controles só por estarem no topo da base
        ordem = rng.permutation(len(t))
        usados = set()
        for i in ordem:
            for j in range(k):
                cand = idx[i, j]
                if cand in usados:
                    continue
                if dist[i, j] > caliper:
                    break
                usados.add(cand)
                pares.append((t.index[i], c.index[cand], dist[i, j]))
                break

    p = pd.DataFrame(pares, columns=["idx_tratado", "idx_controle", "distancia"])
    return df.loc[p.idx_tratado], df.loc[p.idx_controle], p


# ------------------------------------------------------------- estimativa
def bootstrap_ic(a, b, n=1200, seed=42):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    dif = [a[rng.integers(0, len(a), len(a))].mean()
           - b[rng.integers(0, len(b), len(b))].mean() for _ in range(n)]
    return np.percentile(dif, [2.5, 97.5])


def main():
    for pasta in ("figuras", "resultados"):
        os.makedirs(pasta, exist_ok=True)
    df = pd.read_csv("dados/base_campanha.csv")
    tratados_todos = df[df.comunicado == 1]
    controles_todos = df[df.comunicado == 0]

    print("=" * 68)
    print(f"Base: {len(df):,} clientes | comunicados: {df.comunicado.mean():.1%}")

    # 1 ---------------------------------------------------- viés de seleção
    bal_antes = tabela_balanco(tratados_todos, controles_todos)
    print("\n[1] Antes do pareamento — quem foi comunicado já era diferente")
    print(bal_antes.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))

    naive = tratados_todos.gasto_pos.mean() - controles_todos.gasto_pos.mean()
    print(f"\n    Comparação ingênua: R$ {naive:,.2f} por cliente")

    # 2 ------------------------------------------------------------ pareia
    t, c, pares = parear(df)
    taxa = len(t) / len(tratados_todos)
    print(f"\n[2] Pareamento 1:1 | {len(t):,} pares ({taxa:.1%} dos comunicados)")

    bal_depois = tabela_balanco(t, c)
    print("\n[3] Depois do pareamento")
    print(bal_depois.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
    ok = bal_depois.smd.abs().max() < 0.10
    print(f"    |SMD| máximo: {bal_depois.smd.abs().max():.3f} "
          f"({'balanceado' if ok else 'ATENÇÃO: resíduo'})")

    # 3 -------------------------------------------------------- incremental
    inc = t.gasto_pos.mean() - c.gasto_pos.mean()
    lo, hi = bootstrap_ic(t.gasto_pos.values, c.gasto_pos.values)
    u, p = mannwhitneyu(t.gasto_pos, c.gasto_pos, alternative="greater")

    att_real = tratados_todos._efeito_real.mean()
    print(f"\n[4] Incremental pareado: R$ {inc:,.2f} "
          f"(IC95% {lo:,.2f} a {hi:,.2f})")
    print(f"    Mann-Whitney U={u:,.0f}  p={p:.2e}")
    print(f"    Faturamento incremental total: "
          f"R$ {inc * len(tratados_todos) / 1e6:,.2f} mi")
    print(f"\n[5] Auditoria — efeito verdadeiro plantado: R$ {att_real:,.2f}")
    print(f"    Ingênuo   erra em {(naive / att_real - 1) * 100:+.0f}%")
    print(f"    Pareado   erra em {(inc / att_real - 1) * 100:+.0f}%")

    # 4 ---------------------------------------------------- por estratégia
    linhas = []
    for est, g in t.groupby("estrategia"):
        gc = c.loc[pares.set_index("idx_tratado")
                    .loc[g.index, "idx_controle"].values]
        d = g.gasto_pos.mean() - gc.gasto_pos.mean()
        l, h = bootstrap_ic(g.gasto_pos.values, gc.gasto_pos.values, n=600)
        _, pv = mannwhitneyu(g.gasto_pos, gc.gasto_pos, alternative="greater")
        linhas.append({"estrategia": est, "n": len(g), "incremental": d,
                       "ic_baixo": l, "ic_alto": h, "p": pv,
                       "real": g._efeito_real.mean()})
    por_est = pd.DataFrame(linhas).sort_values("incremental", ascending=False)
    print("\n[6] Por estratégia")
    print(por_est.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))

    graficos(df, bal_antes, bal_depois, t, c, por_est, naive, inc, att_real)
    por_est.to_csv("resultados/incremental_por_estrategia.csv", index=False)
    print("\nFiguras e resultados salvos.")


# ---------------------------------------------------------------- gráficos
def graficos(df, antes, depois, t, c, por_est, naive, inc, real):
    # 1. balanceamento
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    y = np.arange(len(COVARIAVEIS))
    ax.scatter(antes.smd.abs(), y, s=70, color=CINZA, label="antes", zorder=3)
    ax.scatter(depois.smd.abs(), y, s=70, color=AMBAR, label="depois", zorder=3)
    for i in y:
        ax.plot([abs(antes.smd[i]), abs(depois.smd[i])], [i, i],
                color="#DCE3E4", lw=2, zorder=1)
    ax.axvline(0.10, color=TEAL, ls="--", lw=1.2)
    ax.text(0.108, 3.35, "limite 0,10", color=TEAL, fontsize=8, va="center")
    ax.set_yticks(y, COVARIAVEIS)
    ax.set_xlabel("|diferença padronizada de médias|")
    ax.set_title("O pareamento resolveu o desequilíbrio", loc="left",
                 fontsize=12, pad=12)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig("figuras/01-balanceamento.png"); plt.close(fig)

    # 2. sobreposição de gasto_pre
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2), sharex=True, sharey=True)
    bins = np.linspace(0, df.gasto_pre.quantile(0.97), 44)
    for ax, (a, b, tit) in zip(axes, [
        (df[df.comunicado == 1], df[df.comunicado == 0], "Antes do pareamento"),
        (t, c, "Depois do pareamento")]):
        ax.hist(a.gasto_pre, bins=bins, color=AMBAR, alpha=.62, label="comunicado")
        ax.hist(b.gasto_pre, bins=bins, color=TEAL, alpha=.55, label="controle")
        ax.set_title(tit, loc="left", fontsize=11)
        ax.set_xlabel("gasto nos 3 meses anteriores (R$)")
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig("figuras/02-vies-selecao.png"); plt.close(fig)

    # 3. naive vs pareado vs verdade
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    vals = [naive, inc, real]
    nomes = ["Comparação\ningênua", "Controle\npareado", "Efeito\nverdadeiro"]
    ax.bar(nomes, vals, color=[CINZA, AMBAR, TEAL], width=.58)
    for i, v in enumerate(vals):
        ax.text(i, v + max(vals) * .03, f"R$ {v:,.0f}", ha="center", fontsize=10)
    ax.set_ylabel("incremental por cliente (R$)")
    ax.set_ylim(0, max(vals) * 1.22)
    ax.set_title("Sem pareamento, o resultado é superestimado",
                 loc="left", fontsize=12, pad=12)
    fig.tight_layout(); fig.savefig("figuras/03-naive-vs-pareado.png"); plt.close(fig)

    # 4. por estratégia
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    y = np.arange(len(por_est))
    ax.barh(y, por_est.incremental, color=AMBAR, height=.55)
    ax.errorbar(por_est.incremental, y,
                xerr=[por_est.incremental - por_est.ic_baixo,
                      por_est.ic_alto - por_est.incremental],
                fmt="none", ecolor=TINTA, elinewidth=1.3, capsize=4)
    ax.set_yticks(y, por_est.estrategia)
    ax.invert_yaxis()
    ax.set_xlabel("incremental por cliente (R$), IC 95%")
    ax.set_title("Nem toda régua entrega o mesmo", loc="left",
                 fontsize=12, pad=12)
    fig.tight_layout(); fig.savefig("figuras/04-por-estrategia.png"); plt.close(fig)


if __name__ == "__main__":
    main()
