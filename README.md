# ДЗ3. Обучение ранжированию

Kaggle: **Zybchenko Egor**

Public Score: **0.62605**

К 67 признакам из задания добавляю 14 нормированных значений внутри запроса и 16 статистик по его кандидатам: среднее и максимум восьми готовых признаков

- CatBoost обучается с `YetiRank`, 600 деревьями, глубиной 6, скоростью обучения 0.05 и `l2_leaf_reg=5`.
- LightGBM обучается с `lambdarank`, 650 деревьями, глубиной 5, до 15 листьев, скоростью обучения 0.04, `min_data_in_leaf=180`, `lambda_l2=8`, `max_bin=127`, `feature_fraction=0.85`. `label_gain=[0, 1, 3, 7]`, `lambdarank_truncation_level=13`.
- Нейросеть на NumPy использует 64 скрытых нейрона, 25 эпох, Adam со скоростью 0.001 и скользящее среднее весов начиная с 500-го шага

Проверено на Python 3.12. Для LightGBM на Debian и Ubuntu нужна библиотека libgomp.so.1, которая поставляется в пакете libgomp1.
Перед объединением скоры каждой модели стандартизируются внутри запроса. Веса ансамбля: 0.60 для CatBoost, 0.20 для LightGBM и 0.20 для нейросети. Seed обучения — 0.


```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
chmod +x solution.py
```

```bash
./solution.py --train --model_file=model.bin PATH-TO-DATA
./solution.py --model_file=model.bin --submission_file=submission.csv PATH-TO-DATA
```

`PATH-TO-DATA` — каталог с `train.csv` и `test.csv`. Первая команда сохраняет все три модели в `model.bin`. Вторая записывает `ObjectId,Score` для всех 94 633 строк теста. Обучение на CPU Ubuntu 24.04 заняло около минуты при пиковом использовании около 1 ГиБ памяти.
