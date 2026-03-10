import wave
import struct
import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt
import time

start_time = time.time()
file_name = "2.wav"

# Чтение WAV-файла
with wave.open(file_name, 'rb') as wf:
    # Получаем характеристики аудиоданных
    num_channels = wf.getnchannels()      # Кол-во каналов (моно)
    sample_width = wf.getsampwidth()      # Размер каждого образца (байт)
    frame_rate = wf.getframerate()        # Частота дискретизации
    num_frames = wf.getnframes()          # Количество кадров (отсчётов)
    raw_data = wf.readframes(num_frames)

# Преобразование сырого массива данных в массив отсчётов
sample_bits = sample_width * 8              # Количество бит на образец
samples = struct.unpack(f'<{num_frames}h', raw_data)  
samples = np.array(samples)

# Задание длительности и количества точек для построения
duration = num_frames / frame_rate               # Общая длительность звука
number_of_samples = int(input("Введите количество отсчётов для визуализации: "))
sampling_points = samples[:number_of_samples]    # Берём первые N отсчётов

# 1.1 - Точечный график дискретных отсчётов
plt.figure(figsize=(12, 6))
plt.plot(np.arange(number_of_samples), sampling_points, '.', markersize=3, label=f'{number_of_samples} отсчётов')
plt.title('График дискретных отсчётов звукового сигнала')
plt.xlabel('Номер отсчёта')
plt.ylabel('Амплитуда')
plt.legend()
plt.grid(True)

# 1.2 - Осциллограмма звукового сигнала
time_axis = np.linspace(0, duration, number_of_samples)
plt.figure(figsize=(12, 6))
plt.plot(time_axis, sampling_points, label='Осциллограмма')
plt.title('Осциллограмма звукового сигнала')
plt.xlabel('Время, секунд')
plt.ylabel('Амплитуда')
plt.legend()
plt.grid(True)

# 1.3 - Спектральный анализ (ДПФ)
fft_result = fftpack.fft(sampling_points)
frequencies = fftpack.fftfreq(number_of_samples, d=1./frame_rate)
real_part_fft = np.real(fft_result)

plt.figure(figsize=(12, 6))
plt.plot(frequencies[:number_of_samples//2], real_part_fft[:number_of_samples//2])
plt.title('Спектральная плотность реального компонента ДПФ')
plt.xlabel('Частота, Гц')
plt.ylabel('Амплитуда')
plt.grid(True)

# 1.4 - Гистограмма распределения отсчётов
bins = int(input("Введите количество бинов для гистограммы: "))
plt.figure(figsize=(12, 6))
plt.hist(sampling_points, bins=bins, density=True, alpha=0.7, color='blue', edgecolor='black')
plt.title('Гистограмма распределения амплитудных значений отсчётов')
plt.xlabel('Амплитуда')
plt.ylabel('Частота появления')
plt.grid(True)

execution_time = time.time() - start_time
print(f"Время выполнения программы: {execution_time:.4f} seconds")
plt.show()