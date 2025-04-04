# Jakub Wojtkowiak, 193546

import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def ema(per, input, i):
    elem = 1 - (2 / (per + 1))
    l, m = 0, 0
    for j in range(per):
        if i - j < 0:
            break
        l += input[i - j] * pow(elem, j)
        m += pow(elem, j)
    return l/m

def macd(input, i):
    return ema(12, input, i) - ema(26, input, i)

def signal(input, i):
    return ema(9, input, i)

def wma(per, input, i):
    s = per*(per+1)/2
    l = 0
    for j in range(i, i-per, -1):
        if j < 0:
            break
        l += input[j]*(per-i+j)
    return l/s

def hma(per, input, i):
    rawHMA = 2*wma(per//2, input, i) - wma(per, input, i)
    rHMAarr = [rawHMA for _ in range(int(math.sqrt(per)))]
    return wma(int(math.sqrt(per)), rHMAarr, int(math.sqrt(per))-1)


def plot_fragment(date1, date2):
    global x, instr_vals, sell_scatter_points, buy_scatter_points, macd_vals, signal_vals
    left, right = 0, 0

    for i, date in enumerate(x):
        if np.datetime_as_string(date) == date1:
            left = i
        elif np.datetime_as_string(date) == date2:
            right = i + 1

    sell_idx = [i for i, d in enumerate(sell_scatter_points[0]) if
                np.datetime64(date1) <= d <= np.datetime64(date2)]
    sells = [[sell_scatter_points[0][x] for x in sell_idx], [sell_scatter_points[1][x] for x in sell_idx]]

    buy_idx = [i for i, d in enumerate(buy_scatter_points[0]) if
               np.datetime64(date1) <= d <= np.datetime64(date2)]
    buys = [[buy_scatter_points[0][x] for x in buy_idx], [buy_scatter_points[1][x] for x in buy_idx]]

    plt.subplot(2, 1, 1)
    plt.plot(x[left:right], instr_vals[left:right])
    plt.scatter(sells[0], [instr_vals[i] for i, m in enumerate(x) if m in sells[0]], color='g', zorder=2, label="Sprzedaż")
    plt.scatter(buys[0], [instr_vals[i] for i, m in enumerate(x) if m in buys[0]], color='r', zorder=2, label="Zakup")
    plt.title("Wykres wartości indeksu WIG20")
    plt.xlabel("Data")
    plt.ylabel("Wartość [zł]")

    plt.subplot(2, 1, 2)
    plt.plot(x[left:right], macd_vals[left:right], zorder=1, label="MACD")
    plt.plot(x[left:right], signal_vals[left:right], zorder=1, label="SIGNAL")
    plt.scatter(sells[0], sells[1], color='g', zorder=2, label="Sprzedaż")
    plt.scatter(buys[0], buys[1], color='r', zorder=2, label="Zakup")
    plt.legend()
    plt.xlabel("Data")
    plt.ylabel("Wartość MACD/SIGNAL")
    plt.title("Wykres MACD oraz SIGNAL")
    plt.show()


data = pd.read_csv("WIG20.csv")
instr_vals = data["Zamkniecie"]

x = np.asarray(data["Data"], dtype='datetime64')
macd_vals, signal_vals = [], []
sell_scatter_points = [[], []]
buy_scatter_points = [[], []]
capital = 1000
value = 0

plt.plot(x, instr_vals)
plt.title("Wykres wartości indeksu WIG20")
plt.xlabel("Data")
plt.ylabel("Wartość [zł]")
plt.show()

print(f"Kup i trzymaj:\n{1000 * instr_vals[0]} zł -> {1000 * instr_vals[999]} zł ||"
      f" {(100 * (1000 * instr_vals[999]) / (1000 * instr_vals[0])) - 100}% zysku")

print("\nMACD:")

for i in range(1000):
    macd_vals.append(macd(instr_vals, i))
    signal_vals.append(signal(macd_vals, i))
    if i >= 37:  # 1st SIGNAL accurate after 26 + 9 iterations
        if macd_vals[i - 1] >= signal_vals[i - 1] and macd_vals[i - 2] < signal_vals[i - 2] and value >= instr_vals[i]:
            print(f"Kupno:\t\t{x[i]} | akcje: {capital} -> {capital + int(value / instr_vals[i])},"
                  f" saldo: {value} -> {round(value - int(value / instr_vals[i]) * instr_vals[i], 2)}")
            capital += int(value / instr_vals[i])
            value = round(value - int(value / instr_vals[i]) * instr_vals[i], 2)
            buy_scatter_points[0].append(x[i])
            buy_scatter_points[1].append(macd_vals[i])
        elif macd_vals[i - 1] <= signal_vals[i - 1] and macd_vals[i - 2] > signal_vals[i - 2] and capital >= 0:
            print(f"Sprzedaż:\t{x[i]} | akcje: {capital} -> 0,"
                  f" saldo: {value} -> {round(value + capital * instr_vals[i], 2)}")
            value = round(value + capital * instr_vals[i], 2)
            capital = 0
            sell_scatter_points[0].append(x[i])
            sell_scatter_points[1].append(macd_vals[i])

print(f"{1000 * instr_vals[0]} zł -> {value + capital * instr_vals[999]} zł ||"
      f" {(100 * (value + capital * instr_vals[999]) / (1000 * instr_vals[0])) - 100}% zysku")


plt.plot(x, macd_vals, zorder=1, label="MACD")
plt.plot(x, signal_vals, zorder=1, label="SIGNAL")
plt.scatter(sell_scatter_points[0], sell_scatter_points[1], color = 'g', zorder=2, label="Sprzedaż")
plt.scatter(buy_scatter_points[0], buy_scatter_points[1], color = 'r', zorder=2, label="Zakup")
plt.legend()
plt.xlabel("Data")
plt.ylabel("Wartość MACD/SIGNAL")
plt.title("Wykres MACD oraz SIGNAL")
plt.show()

#################################################

print("\nHMA:")
hma21_vals, hma50_vals = [], []
hma_sell_scatter_points = [[], []]
hma_buy_scatter_points = [[], []]
capital = 1000
value = 0

for i in range(1000):
    if i >= 20:
        hma21_vals.append(hma(21, instr_vals, i))
    else:
        hma21_vals.append(math.nan)
    if i >= 49:
        hma50_vals.append(hma(50, instr_vals, i))
    else:
        hma50_vals.append(math.nan)
    if i >= 51:
        if hma21_vals[i - 1] >= hma50_vals[i - 1] and hma21_vals[i - 2] < hma50_vals[i - 2] and value >= instr_vals[i]:
            print(f"Kupno:\t\t{x[i]} | akcje: {capital} -> {capital + int(value / instr_vals[i])},"
                  f" saldo: {value} -> {round(value - int(value / instr_vals[i]) * instr_vals[i], 2)}")
            capital += int(value / instr_vals[i])
            value = round(value - int(value / instr_vals[i]) * instr_vals[i], 2)
            hma_buy_scatter_points[0].append(x[i])
            hma_buy_scatter_points[1].append(hma21_vals[i])
        elif hma21_vals[i - 1] <= hma50_vals[i - 1] and hma21_vals[i - 2] > hma50_vals[i - 2] and capital >= 0:
            print(f"Sprzedaż:\t{x[i]} | akcje: {capital} -> 0,"
                  f" saldo: {value} -> {round(value + capital * instr_vals[i], 2)}")
            value = round(value + capital * instr_vals[i], 2)
            capital = 0
            hma_sell_scatter_points[0].append(x[i])
            hma_sell_scatter_points[1].append(hma21_vals[i])

print(f"{1000 * instr_vals[0]} zł -> {value + capital * instr_vals[999]} zł ||"
      f" {(100 * (value + capital * instr_vals[999]) / (1000 * instr_vals[0])) - 100}% zysku")

plt.plot(x, hma21_vals, zorder=1, label="HMA21")
plt.plot(x, hma50_vals, zorder=1, label="HMA50")
plt.scatter(hma_sell_scatter_points[0], hma_sell_scatter_points[1], color = 'g', zorder=2, label="Sprzedaż")
plt.scatter(hma_buy_scatter_points[0], hma_buy_scatter_points[1], color = 'r', zorder=2, label="Zakup")
plt.legend()
plt.xlabel("Data")
plt.ylabel("Wartość HMA")
plt.title("Wykres HMA")
plt.show()

#################################################

plot_fragment("2022-06-20", "2022-12-01")
plot_fragment("2021-09-06", "2022-06-10")