![Классификация Lenta.ru: TF-IDF, BM25 и CatBoost](docs/banner.svg)

# Классификация Lenta.ru: TF-IDF, BM25 и CatBoost

Предсказать рубрику новости и сравнить два подхода: собственное TF-IDF-представление текста и классификацию по соседям из поискового индекса.

## Подход

Первый метод — лемматизация pymorphy3, собственный TF-IDF и LogisticRegression. Второй — три отдельных Lucene BM25-индекса по заголовку, тексту и тегам. Темы и оценки 20 соседей каждого поля образуют 120 признаков для CatBoost.

Весь основной код и результаты экспериментов находятся в одном [project.ipynb](project.ipynb).

## Данные

Lenta.ru: 800 975 строк CSV, из которых 738 973 имеют рубрику. BM25 reference-корпус содержит 618 518 новостей, CatBoost train — 68 725, test — 51 730. TF-IDF обучается на полной train-части с непустыми текстами; словарь содержит 583 280 терминов.

## Результаты

| Метод | Test accuracy | Macro F1 |
|---|---:|---:|
| TF-IDF + LogisticRegression | 0.826619 | 0.524016 |
| BM25 + CatBoost | **0.901257** | **0.723403** |

Используется разбиение 93%/7% с seed=42; редкие классы обрабатываются отдельно. Test одинаков для обоих методов. CatBoost и LogisticRegression обучены на полных train-частях. Точные дубликаты не удаляются, временное разбиение не применяется.

## Запуск

Положите `lenta-ru-news.csv` в `data/raw/`.

```bash
python -m pip install -r requirements.txt
jupyter lab project.ipynb
```

В Jupyter выберите **Restart Kernel and Run All Cells**. Для BM25 нужны Java 21+ и Anserini JAR, поставляемый с Pyserini. CatBoost обучается на GPU. Локальные индексы и соседей можно переиспользовать; при отсутствии индексов ноутбук строит их по полным данным.

Подготовка данных и моделей: [data/README.md](data/README.md).

## Ключевые соображения

- TF=count/length, IDF=log((1+N)/(1+DF)); дополнительного +1 и L2-нормализации нет.
- LogisticRegression: C=1, lbfgs, tol=1e-7, max_iter=2000.
- BM25 использует Java Lucene и три независимых поля.
- Теги — отдельный информативный вход, поэтому результат описывает классификацию по тексту и метаданным.
- Macro F1 показывает качество на редких рубриках при дисбалансе классов.

## Артефакты

`catboost.cbm` — модель; `catboost_predictions.csv` — предсказания на test; `classification_report.json` и `confusion_matrix.csv` — отчёты; `tfidf_metrics.json` — метрики текстового baseline. Большой TF-IDF-классификатор и Lucene-индексы сохраняются локально.

[Метрики](artifacts/full/metrics.json) · [Объём данных](artifacts/full/input_counts.json) · [Сравнение методов](artifacts/full/comparison.png)

`artifacts/full/` содержит результаты эксперимента. Сырые данные, большие веса, индексы и кэши сохраняются в `data/raw/` и `artifacts/local/`; эти каталоги исключены из Git.

## Данные и базовые модели

Датасет: [Lenta.Ru News Dataset](https://github.com/yutkin/Lenta.Ru-News-Dataset).
