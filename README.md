# Deepfake Detection — ML Intensive Yandex Academy Spring 2026, Team 11

Задача: бинарная классификация изображений (настоящее / дипфейк).  

## Что внутри

Ноутбук содержит несколько итераций эксперементов:

- **Кастомная сеть с MultiScale-блоками** — параллельные свёртки 1×1, 3×3, 5×5 с SE-вниманием
- **EfficientNet-B0** — написан вручную, без pretrained-весов
- **DualEfficientNetV2** — ансамбль двух EfficientNet-B0 поверх разных входов

Финальный сабмит — из ансамблевого подхода с Focal Loss.

## Обучение

- Аугментации: albumentations (crop, flip, brightness, noise, compression, dropout)
- MixUp на уровне батчей
- WeightedRandomSampler для балансировки классов
- Mixed Precision (AMP)
- CosineAnnealingWarmRestarts
- Автоматический подбор порога по F1 на валидации

## Метрики

F1-score + ROC-AUC, стратифицированный сплит 90/10.

## Запуск

```bash
# 1. Положить kaggle.json и скачать данные
kaggle competitions download -c ml-intensive-yandex-academy-spring-2026

# 2. Открыть main.ipynb в Google Colab и запустить все ячейки
# Результат сохраняется в submission.csv
```

## Стек

Python, PyTorch, albumentations, scikit-learn, pandas, Google Colab
