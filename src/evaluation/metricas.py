"""
Métricas para evaluar recomendaciones top-k (por defecto k=12).

  recomendados: dict cliente_id -> lista ordenada de ids de producto
  reales:       dict cliente_id -> set de ids comprados en el periodo de evaluación

Reglas:
  - Se promedia solo sobre clientes con al menos una compra real.
  - Un cliente sin recomendaciones cuenta como 0.
  - Ids duplicados en la lista: solo cuenta la primera aparición (se quitan
    antes de cortar a k, así la lista tiene k ids distintos).
  - MAP@k como en la competencia de Kaggle H&M:
      AP = sum_{i=1..k} P(i)·rel(i) / min(|reales|, k)

Uso:
  python src/evaluation/metricas.py   (corre las pruebas y imprime "ok")
"""


def _top_k(lista, k):
    """Quita duplicados (queda la primera aparición) y devuelve los primeros k."""
    return list(dict.fromkeys(lista))[:k]


def _recall(top, reales, k):
    return len(set(top) & reales) / len(reales)


def _precision(top, reales, k):
    return len(set(top) & reales) / k


def _ap(top, reales, k):
    aciertos, suma = 0, 0.0
    for i, item in enumerate(top, 1):
        if item in reales:
            aciertos += 1
            suma += aciertos / i
    return suma / min(len(reales), k)


def _promedio(funcion, recomendados, reales, k):
    """Promedia la métrica por cliente, solo clientes con compras reales."""
    valores = [funcion(_top_k(recomendados.get(c, []), k), r, k)
               for c, r in reales.items() if r]
    return sum(valores) / len(valores) if valores else 0.0


def recall_at_k(recomendados, reales, k=12):
    return _promedio(_recall, recomendados, reales, k)


def precision_at_k(recomendados, reales, k=12):
    return _promedio(_precision, recomendados, reales, k)


def map_at_k(recomendados, reales, k=12):
    return _promedio(_ap, recomendados, reales, k)


def evaluar(recomendados, reales, k=12):
    """Devuelve las tres métricas en un dict."""
    return {
        f"recall@{k}": recall_at_k(recomendados, reales, k),
        f"precision@{k}": precision_at_k(recomendados, reales, k),
        f"map@{k}": map_at_k(recomendados, reales, k),
    }


if __name__ == "__main__":
    def igual(a, b):
        return abs(a - b) < 1e-9

    # Perfecto: acierta todo, en orden -> todo 1.0 (k=2, 2 compras)
    r = evaluar({1: ["a", "b"]}, {1: {"a", "b"}}, k=2)
    assert all(igual(v, 1.0) for v in r.values()), r

    # Cero aciertos -> todo 0.0
    r = evaluar({1: ["x", "y", "z"]}, {1: {"a", "b"}}, k=3)
    assert all(igual(v, 0.0) for v in r.values()), r

    # A mano: aciertos en posición 2 y 4, k=4
    #   recall = 2/2 = 1, precision = 2/4 = 0.5, AP = (1/2 + 2/4) / min(2,4) = 0.5
    r = evaluar({1: ["x", "a", "y", "b"]}, {1: {"a", "b"}}, k=4)
    assert igual(r["recall@4"], 1.0) and igual(r["precision@4"], 0.5) and igual(r["map@4"], 0.5), r

    # Varios clientes, k=12 por defecto: c1 acierta 1/1, c2 sin recomendaciones,
    # c3 sin compras reales (se ignora) -> recall 0.5, precision (1/12)/2, MAP 0.5
    r = evaluar({1: ["a"]}, {1: {"a"}, 2: {"b"}, 3: set()})
    assert igual(r["recall@12"], 0.5) and igual(r["precision@12"], 1 / 24) and igual(r["map@12"], 0.5), r

    # Duplicados: [a, a, b] cuenta como [a, b] -> AP = (1 + 1) / 2 = 1
    assert igual(map_at_k({1: ["a", "a", "b"]}, {1: {"a", "b"}}, k=3), 1.0)

    # Más compras que k: denominador de AP = min(5, 2) = 2 -> AP = 1, recall = 2/5
    r = evaluar({1: ["a", "b"]}, {1: {"a", "b", "c", "d", "e"}}, k=2)
    assert igual(r["map@2"], 1.0) and igual(r["recall@2"], 0.4), r

    print("ok")
