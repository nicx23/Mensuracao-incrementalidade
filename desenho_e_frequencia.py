"""
Dois problemas que aparecem antes e depois da mensuração de incrementalidade.

CASO 2 — Desenho do teste: qual holdout e por quantas semanas, para o
         incremental que se quer detectar ser detectável.
CASO 3 — Cobertura x frequência: com capacidade fixa de disparo, alcançar
         mais gente uma vez ou menos gente mais vezes?

Dados sintéticos. Parâmetros de gasto calibrados em ordem de grandeza
plausível, sem informação de qualquer empresa.
"""

import numpy as np
from scipy.stats import norm

# ------------------------------------------------------------------ caso 2
GASTO_MEDIO, GASTO_DP = 1150.0, 1400.0   # gasto mensal: média e dispersão
ALFA, PODER = 0.05, 0.80


def n_por_grupo(mde_reais, dp=GASTO_DP, alfa=ALFA, poder=PODER):
    """Amostra por braço para detectar uma diferença de médias de `mde_reais`."""
    z_a = norm.ppf(1 - alfa / 2)
    z_b = norm.ppf(poder)
    return int(np.ceil(2 * ((z_a + z_b) * dp / mde_reais) ** 2))


def mde_detectavel(n_hold, n_trat, dp=GASTO_DP, alfa=ALFA, poder=PODER):
    """Menor efeito detectável dado o tamanho dos dois braços."""
    z = norm.ppf(1 - alfa / 2) + norm.ppf(poder)
    return z * dp * np.sqrt(1 / n_hold + 1 / n_trat)


def caso2(base=2_000_000):
    print("CASO 2 — quanto o holdout escolhido consegue enxergar")
    print(f"base eleg. {base:,} | gasto medio R$ {GASTO_MEDIO:,.0f} | "
          f"dp R$ {GASTO_DP:,.0f} | alfa {ALFA} | poder {PODER:.0%}\n")
    print(f"{'holdout':>9} {'n controle':>12} {'MDE (R$)':>11} {'% do gasto':>12}")
    linhas = []
    for h in [0.01, 0.02, 0.05, 0.10, 0.15]:
        nh = int(base * h)
        nt = base - nh
        m = mde_detectavel(nh, nt)
        linhas.append((h, nh, m, m / GASTO_MEDIO))
        print(f"{h:>8.0%} {nh:>12,} {m:>11.2f} {m/GASTO_MEDIO:>11.1%}")

    print("\n  o caminho inverso — holdout necessário por efeito alvo:")
    for mde in [5, 10, 20, 40]:
        n = n_por_grupo(mde)
        print(f"    detectar R$ {mde:>3} ({mde/GASTO_MEDIO:>4.1%}) "
              f"-> {n:>9,} no controle = {n/base:>6.1%} da base")
    return linhas


# ------------------------------------------------------------------ caso 3
def resposta(freq, teto=1.0, k=0.55, fadiga=0.018):
    """
    Resposta por pessoa a n comunicações no mês.
    Ganho decrescente (saturação) menos um termo de fadiga que cresce
    com o quadrado da frequência — a partir de certo ponto, insistir custa.
    """
    return teto * (1 - np.exp(-k * freq)) - fadiga * freq ** 2


def caso3():
    print("\n\nCASO 3 — cobertura x frequência")
    base = 12_000_000        # base elegível
    capacidade = 18_000_000  # disparos disponíveis no mês
    print(f"base {base:,} | capacidade {capacidade:,} disparos/mês\n")
    print(f"{'freq':>5} {'cobertura':>11} {'resp/pessoa':>13} {'resultado':>14}")

    melhor, linhas = None, []
    for freq in range(1, 11):
        cobertura = min(capacidade / (freq * base), 1.0)
        r = resposta(freq)
        total = cobertura * base * r
        linhas.append((freq, cobertura, r, total))
        print(f"{freq:>5} {cobertura:>10.1%} {r:>13.3f} {total:>14,.0f}")
        if melhor is None or total > melhor[3]:
            melhor = (freq, cobertura, r, total)

    f, cob, r, tot = melhor
    f1 = linhas[0]
    print(f"\n  ótimo: {f} comunicações para {cob:.0%} da base")
    print(f"  contra alcance máximo (1x para {f1[1]:.0%}): {tot/f1[3]-1:+.0%}")
    neg = [l for l in linhas if l[2] < 0]
    if neg:
        print(f"  a partir de {neg[0][0]}x a resposta por pessoa fica negativa")
    return linhas, melhor


if __name__ == "__main__":
    caso2()
    caso3()
