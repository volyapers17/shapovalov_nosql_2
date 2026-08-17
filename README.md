# Семантичний пошук за науковими статтями

## 1. Мета роботи

Побудувати повний пошуковий pipeline для 10 000 статей arXiv: підготувати сирі дані, перетворити назви й анотації на вектори, завантажити їх у Pinecone, виконати пошук із фільтрами, порівняти стратегії chunking та об'єднати BM25 із векторним пошуком через RRF.

Головна різниця між методами така:

- **BM25** шукає збіги слів. Він сильний для точних термінів, скорочень та імен.
- **Векторний пошук** порівнює зміст текстів. Він може знайти релевантний документ, навіть коли запит сформульовано іншими словами.
- **Гібридний пошук** бере кандидатів з обох списків і об'єднує їх ранги.

## 2. Структура проєкту

```text
.
├── .env.example
├── .gitignore
├── requirements.txt
├── data/                         # генерується локально, не додається в Git
│   └── arxiv_subset.parquet
├── embeddings/                   # генерується локально, не додається в Git
│   └── embeddings.npy
└── scripts/
    ├── 01_prepare_data.py
    ├── 02_embed.py
    ├── 03_load_to_pinecone.py
    ├── 04_search.py
    ├── 05_chunking.py
    └── 06_hybrid_search.py
```

Файл `.env` також існує локально, але навмисно прихований через `.gitignore`, тому API-ключ не потрапляє в репозиторій.

## 3. Запуск

Скрипти потрібно запускати **тільки в порядку від 01 до 06**, тому що кожен наступний використовує результат попереднього.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# вставити власний PINECONE_API_KEY у .env

python scripts/01_prepare_data.py \
  --input /path/to/arxiv-metadata-oai-snapshot.json
python scripts/02_embed.py
python scripts/03_load_to_pinecone.py
python scripts/04_search.py
python scripts/05_chunking.py
python scripts/06_hybrid_search.py
```

## 4. Інструменти та бібліотеки

| Інструмент | Для чого використаний | Чому саме він |
|---|---|---|
| `pandas` | таблиця статей, читання і запис Parquet | зручно працювати з колонками, фільтрами та пропусками |
| `pyarrow` | Parquet-двигун для pandas | Parquet компактніший і швидше читається, ніж повторне читання JSONL на 5.1 GB |
| `NumPy` | масив ембеддингів і локальні метрики | швидкі векторні обчислення та формат `.npy` |
| `sentence-transformers` | завантаження SPECTER2 та batch-кодування | дає простий метод `encode`, нормалізацію і batch-обробку |
| `Pinecone` | хмарний векторний індекс, metadata filters | не потрібно самостійно розгортати та обслуговувати сервер |
| `rank-bm25` | локальний повнотекстовий пошук | готова зрозуміла реалізація BM25 без окремої пошукової системи |
| `tqdm` | progress bar | видно, що довга операція працює і скільки залишилося |
| `python-dotenv` | читання API-ключа з `.env` | секрет не записується в код |
| `kaggle` | завантаження датасету | офіційний CLI джерела датасету |

## 5. Частина 1 — дані та вибір інструментів

### 5.1. Підготовка даних

JSONL читається рядок за рядком, тому 5.1 GB не завантажуються в оперативну пам'ять. Для цієї роботи береться кожен 250-й запис до отримання 10 000 валідних статей. Це важливо, бо файл упорядкований приблизно за часом: перші 10 000 рядків дали б переважно старі статті, і фільтр за останні п'ять років не мав би сенсу.

Зберігаються `id`, `title`, `abstract`, `authors`, рік першої версії та перша категорія. Повторювані пробіли й переноси рядків прибираються.

Скорочений вивід `01_prepare_data.py`:

```text
Завантажено статей: 10000
Роки: 2007–2025
Топ категорій:
cs.CV  575
cs.LG  482
quant-ph 424
hep-ph 345
cs.CL  272
Збережено в data/arxiv_subset.parquet
```

### 5.2. Pinecone, Qdrant і Chroma

| Система | Розгортання та ліцензія | Продуктивність і типовий сценарій |
|---|---|---|
| Pinecone | керований комерційний хмарний сервіс; сервери обслуговує Pinecone | обрав би для production, коли потрібні автоматичне масштабування, доступність і мінімум DevOps |
| Qdrant | open source Apache 2.0, self-hosted, Managed/Hybrid/Private Cloud | обрав би для production із контролем над інфраструктурою, даними, HNSW та складною фільтрацією |
| Chroma | open source Apache 2.0, локальний embedded/server режим і Chroma Cloud | обрав би для ноутбука, прототипу або невеликого RAG, де найважливіший швидкий старт |

Не можна чесно назвати одну базу завжди найшвидшою: результат залежить від кількості векторів, dimension, індексу, recall, фільтрів і обладнання. Pinecone зручний керованим масштабуванням; Qdrant дає більше контролю; Chroma має найнижчий поріг входу. Джерела: [Pinecone index docs](https://docs.pinecone.io/reference/api/2026-04/control-plane/create_index), [Qdrant deployment overview](https://qdrant.tech/documentation/overview/), [Qdrant Apache 2.0](https://github.com/qdrant/qdrant), [Chroma Cloud та Apache 2.0](https://docs.trychroma.com/cloud), [Chroma performance](https://docs.trychroma.com/production/administration/performance).

### 5.3. Чому SPECTER2, а не all-MiniLM-L6-v2

`all-MiniLM-L6-v2` — компактна універсальна модель для речень і коротких текстів. Вона швидша й має лише 384 виміри, але не спеціалізується на наукових статтях.

SPECTER2 навчався на понад 6 млн citation triplets із 23 наукових галузей. Цитування є корисним сигналом: якщо одна стаття посилається на іншу, вони часто пов'язані змістом, навіть коли використовують різні слова. У [картці `allenai/specter2_base`](https://huggingface.co/allenai/specter2_base) сказано, що модель створює embeddings для scientific tasks і підтримує формати classification, regression, proximity/retrieval та ad-hoc search. Тому вона логічніша для arXiv.

Формат `title + " [SEP] " + abstract` повторює навчальний формат моделі. Ліміт становить 512 токенів. Важливий нюанс: картка рекомендує task-specific adapters для найкращої якості. У цій роботі використано саме `SentenceTransformer("allenai/specter2_base")`, як вимагає умова; бібліотека автоматично додає mean pooling. У production я перевірив би proximity adapter для документів і ad-hoc adapter для коротких запитів.

### 5.4. Метрика

Актуальна картка `specter2_base` не задає одну обов'язкову similarity metric; вона описує моделі й task-specific adapters. Тому твердження, що в картці прямо рекомендовано cosine, було б неточним. У цій роботі cosine обрано усвідомлено, бо всі vectors нормалізуються до довжини 1. Індекс Pinecone створюється з тією самою метрикою, інакше спосіб ранжування міг би не відповідати підготовці embeddings.

Скорочений вивід `02_embed.py`:

```text
Оброблено текстів: 10000
Розмірність ембеддингів: 768
Норма першого ембеддингу: 1.000000
Збережено в embeddings/embeddings.npy
```

Перевірка всього масиву: shape `(10000, 768)`, type `float32`, мінімальна і максимальна норми `0.9999997` та `1.0000002`.

### 5.5. Чому cosine дорівнює dot product після нормалізації

Косинусна схожість визначається так:

```text
cos(x, y) = (x · y) / (||x|| ||y||)
```

Після нормалізації `||x|| = ||y|| = 1`, тому знаменник дорівнює 1:

```text
cos(x, y) = x · y
```

Отже, значення й порядок документів однакові. Невелика різниця можлива лише через похибку float32.

## 6. Частина 2 — Pinecone і метадані

Створено serverless-індекс `arxiv-papers`: dimension `768`, metric `cosine`, AWS `us-east-1`. Вектори завантажуються по 200. ID має вигляд `paper_0`, а metadata містять arXiv ID, title, скорочений abstract, authors, year і category.

Abstract обрізається до 500 символів, бо Pinecone має ліміт filterable metadata 40 KB на документ. Повний текст уже є в Parquet, тому після пошуку він дістається за `arxiv_id`. Це не дублює великі тексти у векторній базі. [Офіційні ліміти Pinecone](https://docs.pinecone.io/reference/api/database-limits).

Вивід `03_load_to_pinecone.py`:

```text
Створюємо індекс arxiv-papers...
Завантажуємо вектори: 100% | 50/50
Загальна кількість векторів в індексі: 10000
```

## 7. Частина 3 — семантичний пошук і метрики

### 7.1. Результати Pinecone

Для `teaching machines to recognize objects in pictures` топ результатів:

```text
1. Grad-CAM++ is Equivalent to Grad-CAM With Positive Gradients (cs.CV, 2022)
2. Enhancing Object Detection in Ancient Documents... (cs.CV, 2023)
3. Distilling Knowledge from CNN-Transformer Models... (cs.CV, 2023)
4. Hierarchical Explanations for Video Action Recognition (cs.CV, 2023)
5. Teaching Humans Subtle Differences with DIFFusion (cs.CV, 2025)
```

Для `reinforcement learning agents and rewards` із фільтром `cs.LG`, 2021–2025:

```text
1. Unveiling the Significance of Toddler-Inspired Reward Transition... (2024)
2. Designing an efficient and equitable humanitarian supply chain... (2025)
3. Self-Composing Policies for Scalable Continual Reinforcement Learning (2025)
4. Opinion-Guided Reinforcement Learning (2024)
5. An Analysis of Reinforcement Learning for Malaria Control (2021)
```

Фільтр змінює не зміст запиту, а множину дозволених кандидатів. Тому перший список концентрується на computer vision, другий — лише на нових `cs.LG`, а запит із `year <= 2015` повернув старі роботи, наприклад *Dependency-based Convolutional Neural Networks for Sentence Embedding* (2015).

### 7.2. Cosine, dot і L2

Фактичні top-5 для cosine, dot і L2 збіглися. Перші три статті:

```text
1. Grad-CAM++ is Equivalent to Grad-CAM With Positive Gradients
2. Enhancing Object Detection in Ancient Documents...
3. Distilling Knowledge from CNN-Transformer Models...
Cosine == dot top-5: True
Cosine == L2 top-5: True
```

Cosine і dot збігаються через одиничні норми. L2 теж дає той самий порядок для одиничних векторів, хоча числа score інші. Це випливає з формули в розділі 10.3.

Якби embeddings не нормалізувалися, dot product залежав би і від напряму, і від довжини vector. Документ із великою нормою міг би отримати високий dot score без кращої семантичної близькості. Cosine прибирає вплив довжини. L2 без нормалізації також формує інший порядок.

## 8. Частина 4 — chunking

Взято 30 статей із найдовшими abstracts.

- **Fixed-size:** 120 слів, overlap 25. Межа проста, але може розрізати речення.
- **Sentence-aware:** речення додаються цілими, поки chunk не наблизиться до 120 слів. Це проста семантична стратегія без додаткової ML-моделі.

Вивід `05_chunking.py`:

```text
Fixed-size чанків: 120
Sentence-aware чанків: 95
Векторів у arxiv-chunks-fixed: 120
Векторів у arxiv-chunks-semantic: 95

Запит: methods for analyzing human language
Fixed top-1: Mood of India During Covid-19... (chunk 0, score близько 0.786)
Sentence-aware top-1: Mood of India During Covid-19... (chunk 0, score близько 0.786)
```

Sentence-aware chunks осмисленіші для читача, бо думка не обривається посеред речення. Fixed-size у фактичному виводі має фрагменти, які починаються зі слів `with respect to...` або завершуються уривком; embedding такого chunk отримує неповний контекст.

Overlap зменшує ризик втратити думку на межі двох fixed chunks. Чим більший overlap, тим більше chunks, embeddings, місця та часу пошуку. Надто малий overlap може розділити важливий вислів; надто великий створює багато майже однакових результатів. У роботі `25/120 ≈ 21%` — помірне перекриття.

## 9. Частина 5 — BM25 + vector + RRF

BM25 будується локально за `title + abstract`. Для кожного запиту беремо top-10 BM25 і top-10 Pinecone. RRF не порівнює несумісні BM25 та cosine scores, а використовує тільки позиції:

```text
RRF(document) = Σ 1 / (k + rank_i(document))
```

У роботі `k = 60`.

### 9.1. Порівняльна таблиця: 3 запити × 3 методи

| Запит | BM25 top-1 | Vector top-1 | Hybrid top-1 | Висновок |
|---|---|---|---|---|
| `BERT fine-tuning` | *DPBERT: Efficient Inference for BERT...* | *Unveiling... Reward Transition...* | *DPBERT...* | BM25 найкращий: точні слова BERT мають вирішальне значення |
| `Yann LeCun convolutional networks` | *Detecting Cyberattacks... Using CNN* | *Variations on the Chebyshev-Lagrange Activation Function* | *Detecting Cyberattacks...* | у вибірці немає достатнього точного збігу автора; BM25 краще утримує термін CNN |
| `making computers understand human emotions from text` | *Multimodal Sentiment Analysis...* | *Multimodal Sentiment Analysis...* | *Multimodal Sentiment Analysis...* | обидва методи погодилися; hybrid подвоїв підтримку сильного документа |

Скорочений вивід `06_hybrid_search.py` для перефразування:

```text
BM25:
1. Multimodal Sentiment Analysis To Explore the Structure of Emotions
2. The Language of Interoception...

Vector:
1. Multimodal Sentiment Analysis To Explore the Structure of Emotions
2. Generating Emotionally Aligned Responses in Dialogues...

Hybrid RRF:
1. Multimodal Sentiment Analysis...          RRF=0.032787
2. Generating Emotionally Aligned Responses... RRF=0.031754
3. The Language of Interoception...          RRF=0.030622
```

Немає методу, який чесно перемагає на кожному запиті. У цьому запуску BM25 кращий для точного `BERT fine-tuning`; vector корисний для semantic paraphrase; hybrid найстабільніший на різних типах запитів, бо не відкидає сильних кандидатів жодного джерела. Для короткого запиту BERT базовий SPECTER2 без ad-hoc adapter дав слабкі vector results — це реальне обмеження, а не причина приховувати результат.

У фактичному hybrid top-5 не було документа, якого немає в top-5 хоча б одного окремого методу. Але RRF отримує top-10, тому в іншому запиті документ на місцях 6–10 в обох списках може піднятися в hybrid top-5: він отримає внесок двічі.

При `k=1` різниця між першим і десятим місцем дуже велика: `1/2` проти `1/11`. Видача сильно залежить від top-1 кожного методу. При `k=60` це `1/61` проти `1/70`, тому об'єднання м'якше, а документ, який зустрічається в обох списках, отримує стабільну перевагу.

## 10. Частина 6 — аналіз

### 10.1. Семантичний пошук проти BM25

BM25 варто обирати для рідкісного точного терміна, назви методу, коду, абревіатури або автора. У нашому `BERT fine-tuning` він одразу знайшов DPBERT і статтю про fine-tuning. Semantic search потрібен для опису ідеї звичайними словами, синонімів і перефразувань. Приклад `making computers understand human emotions from text` пов'язується з sentiment та emotionally aware systems, хоча запит не містить точного терміна `sentiment analysis`. На практиці hybrid є безпечним default, але його якість усе одно треба оцінювати на розмічених запитах.

### 10.2. Вплив розміру chunk

Chunk на 10–15 слів часто не містить завершеної думки: embedding стає неоднозначним, а кількість vectors різко зростає. Chunk на 500+ слів змішує кілька тем, важливий фрагмент «розчиняється», а модель обрізає все після 512 tokens. Єдиного оптимуму немає. Він залежить від моделі, довжини документів і типу запиту. Для abstracts у цій роботі 120 слів з overlap 25 — зрозуміла стартова точка; для книг або коду потрібен окремий експеримент з recall@k.

### 10.3. Невідповідна метрика

Для одиничних vectors `||x|| = ||y|| = 1`:

```text
||x - y||² = (x - y) · (x - y)
           = ||x||² + ||y||² - 2(x · y)
           = 2 - 2 cos(x, y)
```

Тому мінімізація Euclidean distance еквівалентна максимізації cosine: top-K буде тим самим. Отже, `euclidean` для наших нормалізованих vectors не зламав би математичний порядок, але score мав би інший масштаб, а конфігурація не пояснювала б прямо вибір моделі. Для ненормалізованих vectors еквівалентність зникає.

### 10.4. Pinecone Starter і 10 млн статей

За [актуальними лімітами Pinecone](https://docs.pinecone.io/reference/api/database-limits) Starter має до 5 serverless indexes, 2 GB storage, 100 namespaces, 1 млн read units та 2 млн write units на місяць; region обмежений AWS `us-east-1`, backups недоступні. Робота використовує 3 indexes і легко вкладається в квоту.

Для 10 млн vectors лише float32-матриця `10 000 000 × 768 × 4` займає приблизно 30.7 GB без metadata та службового індексу, тому Starter не підходить. Я перейшов би на платний план або self-hosted Qdrant, генерував embeddings розподілено, завантажував через bulk import, повні тексти тримав в object storage, а vectors розділяв би логічно за галуззю/роком. Якість перевіряв би на окремому наборі запитів через recall@k та nDCG, а не лише візуально.

## 11. Підсумок

У роботі отримано повний відтворюваний pipeline: 10 000 очищених статей → 768-dimensional normalized embeddings → 10 000 vectors у Pinecone → semantic і filtered search → два chunk indexes → BM25 + RRF. Експеримент показав не «абсолютного переможця», а практичне правило: точні слова краще обробляє BM25, перефразування — embeddings, а RRF робить систему стійкішою до обох типів запитів.
