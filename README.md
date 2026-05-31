# License Plate Detection & Recognition Pipeline
End-to-end система детекции и распознавания автомобильных номерных знаков. 
Разработана в рамках курса *"Проектирование архитектур нейронных систем"* (ИТМО, 2 семестр)

## Overview
Пайплайн состоит из двух этапов:
1. **Детекция**: YOLOv8 (Ultralytics) настраивается на поиск области номерного знака.
2. **Распознавание**: CRNN (CNN Backbone + BiLSTM + CTC Loss) преобразует кроп номера в текстовую строку.

Архитектура распознавателя реализована на основе базового кода семинара, адаптирована под российский датасет номеров и дополнена инфраструктурой для проведения экспериментов.

ITMO-License-plate/
├── train_detect.py # Скрипт обучения детектора (YOLOv8)
├── license_plate_config.yaml # Конфигурация датасета для YOLO
├── train_crnn.py # Базовое обучение CRNN (32×100, const LR)
├── train_exp1_resize.py # Эксперимент 1: увеличенный вход (64×200)
├── train_exp2_scheduler.py # Эксперимент 2: LR Scheduler (ReduceLROnPlateau)
├── prepare_crnn_data.py # Подготовка мини-датасета и алфавита
├── make_plots.py # Генерация графиков обучения
├── requirements.txt # Зависимости проекта
├── data/ # Датасет детекции (images/, labels/)
├── autoriaNumberplateOcrRu/ # Датасет распознавания
│ └── mini_dataset/ # Подвыборка для экспериментов
│ ├── train/val/test/ # img/ + ann/
│ ├── alphabet.txt # Словарь символов (+ <blank>)
│ └── labels.txt # Карта соответствий image → text
├── results/ # Результаты детекции
│ ├── training/ # Чекпоинты YOLO, логи, графики
│ └── evaluation/ # Метрики, визуализации предсказаний
└── README.md


## Installation
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

pip install -r requirements.txt
```
## Quick start
### Детекция (YOLOv8)
```bash
# Обучение (30 эпох, CPU/GPU)
python train_detect.py --all --epochs 30 --device cpu

# Оценка и анализ ошибок
python train_detect.py --eval --analyze
```
### Подготовка данных для CRNN
```bash
python prepare_crnn_data.py
```
### Распознавание (CRNN)
```bash
# Baseline
python train_crnn.py

# Эксперименты
python train_exp1_resize.py
python train_exp2_scheduler.py
```
## Experiments & Results

| Эксперимент | Конфигурация | CER (val) ↓ | Exact Match Acc ↑ |
|:------------|:-------------|:-----------:|:-----------------:|
| **Baseline** | `32×100`, `lr=1e-3`, `Adam`, 20 эпох | `0.063` | `0.594` |
| **Exp 1**    | `64×200`, `lr=1e-3`, `Adam`, 20 эпох | `0.205` | `0.178` |
| **Exp 2**    | `32×100` + `ReduceLROnPlateau`, 20 эпох | `0.129` | `0.308` |

