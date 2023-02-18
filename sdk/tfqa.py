import math
import threading
import backtrader as bt
from typing import Mapping, Sequence
import numpy as np
import pandas as pd
import random
from faker import Faker
import time
from typing import Callable, Optional, Union
import websocket
import json
import argparse

parser = argparse.ArgumentParser(description='')

parser.add_argument('--url', type=str,
                    default="", help='server addr to convey trained payload')
parser.add_argument('--token', type=str,
                    default="", help='token')
args = parser.parse_args()

fake = Faker()
local = False

COL = 1
ROW = 0


def noise(x, snr) -> list:
    len_x = len(x)
    s = np.sum(np.power(x, 2)) / len_x
    n = s / (np.power(10, snr / 10))
    noise = np.random.randn(len_x) * np.sqrt(n)
    return list(x + noise)


def random_price(
        _global: bool,
        prices: list[float] | float,
        term: list[int] | int,
        rate: list[float] | float,
        sigma: list[float] | float,
        nper_per_year: list[float] | float,
        codes: list[str],
        random_value: bool = True) -> Callable[[Optional[bool], Optional[list[float]]], Union[float, list[float]]]:
    idx: int = 0
    stocks_num: int = len(codes)
    stock: int = 0
    scores: list[list[float]] = []
    isInc = False
    if not _global:
        if random_value:
            scores = [random_stock_values(
                prices[i], term[i], rate[i], sigma[i], nper_per_year[i]) for i in range(len(codes))]
        else:
            scores = [[0+prices[0] for _ in range(term*nper_per_year)]
                      for _ in range(len(codes))]
    else:
        if random_value:
            scores = [noise(random_stock_values(
                prices, term, rate, sigma, nper_per_year), 50) for _ in range(len(codes))]
        else:
            scores = [[0+prices for _ in range(term*nper_per_year)]
                      for _ in range(len(codes))]

    def random_price_closure(
            getter: bool | None = ...,
            setter: list[float] | None = ...,
            inc: bool | None = ...):
        nonlocal idx, stock, scores, isInc
        if inc != Ellipsis and inc != None:
            isInc = inc
        if getter != Ellipsis and getter != None:
            return [scores[s][idx - 1 if idx >= 1 else 0] for s in range(stocks_num)]
        if setter != Ellipsis and setter != None:
            for s in range(stocks_num):
                scores[s][idx] = setter[s]
                # if not isInc else setter[i] + \
                #     round((random.random() - 0.5) * random.randint(12, 20), 5)
            return
        cur_stock = stock
        cur_idx = idx
        stock = (stock + 1) % stocks_num  # next_stock
        if stock == 0:
            idx = (idx + 1) % len(scores[cur_stock])  # next_idx
        return scores[cur_stock][cur_idx]
    return random_price_closure


def random_kline_data(ts=time.time()*1000, baseValue=3000, dataSize=800):
    ts = math.floor(ts/60/1000)*60*1000
    prices = []
    dataList = []
    for _ in range(dataSize):
        baseValue += random.random() * 20 - 10
        for _ in range(4):
            prices.append((random.random() - 0.5) * 12 + baseValue)
        prices.sort()
        openIdx = round(random.random() * 3)
        closeIdx = round(random.random() * 2)
        if closeIdx == openIdx:
            closeIdx += 1
        volume = random.random() * 50 + 10
        kline = [
            prices[openIdx],
            prices[0],
            prices[3],
            prices[closeIdx],
            volume,
            ts
        ]
        ts -= 60*1000
        kline.append(((kline[0] + kline[3] +
                       kline[2] + kline[1]) / 4) * volume)
        dataList.insert(0, kline)
    return dataList


def set_local():
    global local
    local = True


def set_remote():
    global local
    local = False


class TFQA(object):
    def __init__(self) -> None:
        pass


class KLineData:
    OPEN = 1
    TIMESTAMPLE = 2
    CLOSE = 3
    LOW = 4
    HIGH = 5
    VOLUME = 6
    TURNOVER = 7


def read_csv(
        filepath: str,
        parse_dates: bool | list[int] | list[str] | Sequence[Sequence[int]] | Mapping[str, Sequence[int | str]]) -> pd.DataFrame:
    if local:
        return pd.read_csv(filepath, parse_dates=parse_dates)
    return pd.DataFrame()


class Module(object):
    def __init__(self, cash: float) -> None:
        self.cerebro = bt.Cerebro()
        self.cerebro.broker.setcash(cash)

    def forward(self) -> None:
        pass

    def Sequential(self, strategies: list = ...):
        pass


def fmt2timestamp(fmt: str) -> int:
    return int(time.mktime(time.strptime(fmt, "%Y-%m-%d")))


def timestamp2fmt(timestamp: int) -> str:
    return time.strftime("%Y-%m-%d", time.localtime(timestamp))


def random_float(base: float, codes: list) -> Callable[[Optional[int], Optional[bool]], Union[float, list[float], None]]:
    scores: list[float] = [0+base for _ in range(len(codes))]
    idx: int = 0
    isInc: bool = True

    def random_float_closure(
            score: int | None = ...,
            getter: bool | None = ...,
            setter: list[int] | None = ...,
            inc: bool | None = ...):
        nonlocal scores, idx, isInc
        if inc != Ellipsis and inc != None:
            isInc = inc
        if getter != Ellipsis and getter != None:
            return scores
        if setter != Ellipsis and setter != None:
            scores = [v for v in setter]
            return
        if score != Ellipsis and score != None:
            scores[idx] = score
        idx = (idx + 1) % len(scores)
        if isInc:
            scores[idx] += round((random.random() - 0.5)
                                 * random.randint(12, 20), 5)
        return scores[idx]
    return random_float_closure


def random_time(start: str, num: int, step: int = 60*60*24) -> Callable[[], str]:
    idx: int = 0

    def random_time_closure():
        nonlocal idx, start
        idx = (idx + 1) % num
        if idx == 0:
            start = timestamp2fmt(fmt2timestamp(start) + step)
        return start
    return random_time_closure


def random_code(fmt: str, num: int) -> Callable[[], str]:
    codes = {idx: fake.last_name() for idx in range(num)}
    idx = 0

    def random_code_closure():
        nonlocal idx, codes
        idx = (idx + 1) % num
        return codes[idx]
    return random_code_closure


def bind(func: Callable[..., any], *argv: tuple[tuple, ...]) -> Callable[..., any]:
    def super_func() -> Callable[..., any]:
        return func(*argv)
    return super_func


class DataLoader:
    _data: list
    _cur_data: dict[str, any]
    _callbacks: dict[str, Callable[..., any]]

    def __init__(
            self,
            fileds: list[str],
            basevalue: float | None = ...,
            epoch: int = 1,
            num: int = 1) -> None:

        if len(fileds) <= 0:
            print("argc no match")
            return
        self.basevalue = (lambda v: v if v != None and v !=
                          Ellipsis else 0)(basevalue)
        self._data = []
        self.fields = fileds
        self.epoch = epoch
        self.cur = 0
        self.num = num

    def insert(self, data: list) -> None:
        self._data.append(data)

    def set_callback(self, **callbacks: dict[str, Callable[..., any]]) -> None:
        self._callbacks = callbacks

    def preprocess(self) -> None:
        pass

    def pastprocess(self) -> None:
        pass

    def shuffle(self) -> None:
        self._cur_data = {field: (lambda v: None if not v in self._callbacks.keys(
        ) else self._callbacks[v]())(field) for field in self.fields}

    def forward(self) -> None:
        pass

    def backward(self) -> None:
        pass

    def cvt(self) -> None:
        self._data.append([self._cur_data[key]
                          for key in self._cur_data.keys()])

    def datasets(self) -> pd.DataFrame:
        self.preprocess()
        for _ in range(self.epoch):
            for _ in range(self.num):
                self.forward()
                self.shuffle()
                self.backward()
                self.cvt()
            self.cur += 1
        self.pastprocess()
        return pd.DataFrame(data=self._data, columns=self.fields)

    def format(self) -> str:
        global COL, ROW
        data = pd.DataFrame(data=self._data, columns=self.fields)
        dic = {}
        for stock in data['sec_code'].unique():
            dic[stock] = []
            df = data[data['sec_code'] == stock].drop('sec_code', axis=COL)
            for _, row in df.iterrows():
                dic[stock].append({"open": row['open'], "low": row['low'], "high": row['high'],
                                   "close": row['close'], "volume": row['volume'], "timestamp": fmt2timestamp(row['datetime']) * 1000})
        return json.dumps(dic)


class BaseDataLoader(DataLoader):
    def __init__(
            self,
            fileds: list[str],
            basevalue: float | None = ...,
            epoch: int = 1,
            num: int = 1,
            start: str = '2019-01-01') -> None:
        self.start = start
        super().__init__(fileds, basevalue, epoch, num)

    def preprocess(self) -> None:
        code_handle = random_code('', self.num)
        codes = [code_handle() for _ in range(self.num)]
        open_handle = random_float(self.basevalue, codes)
        close_handle = random_float(self.basevalue, codes)
        time_handle = random_time(
            self.start, self.num)
        self.set_callback(datetime=time_handle, sec_code=code_handle,
                          open=open_handle, close=close_handle)

    def forward(self) -> None:
        if not self.cur == 0:
            close_value: list[float] = self._callbacks['close'](
                None, True, None)
            self._callbacks['open'](None, None, close_value, False)

    def backward(self) -> None:
        self._cur_data['volume'] = round(random.random() * 50 + 10)
        prices: list[float] = []
        for _ in range(4):
            prices.append(round((random.random() - 0.5) *
                          12 + self._cur_data['open'], 5))
        prices.sort()
        min_val: float = min(self._cur_data['open'], self._cur_data['close'])
        max_val: float = max(self._cur_data['open'], self._cur_data['close'])
        if prices[0] >= min_val:
            self._cur_data['low'] = min_val - 1
        else:
            self._cur_data['low'] = prices[0]
        if prices[3] <= max_val:
            self._cur_data['high'] = max_val + 1
        else:
            self._cur_data['high'] = prices[3]


class SingleDataLoader(BaseDataLoader):
    def __init__(self, fileds: list[str], basevalue: float | None = ..., epoch: int = 1, start: str = '2019-01-01') -> None:
        super().__init__(fileds, basevalue, epoch, 1, start)


class MultiDataLoader(BaseDataLoader):
    def __init__(self, fileds: list[str], basevalue: float | None = ..., epoch: int = 1, num: int = 1, start: str = '2019-01-01') -> None:
        super().__init__(fileds, basevalue, epoch, num, start)


class NormDataLoader(BaseDataLoader):
    def __init__(
            self,
            fileds: list[str],
            term: int,
            rate: float,
            sigma: float,
            nper_per_year: int,
            price: float | None = ...,
            epoch: int = 1,
            num: int = 1,
            start: str = '2019-01-01',) -> None:
        super().__init__(fileds, price, epoch, num, start)
        self.term = term
        self.rate = rate
        self.sigma = sigma
        self.nper_per_year = nper_per_year

    def preprocess(self) -> None:
        code_handle = random_code('', self.num)
        codes = [code_handle() for _ in range(self.num)]
        open_handle = random_price(
            True, self.basevalue, self.term, self.rate, self.sigma, self.nper_per_year, codes, False)
        close_handle = random_price(
            True, self.basevalue, self.term, self.rate, self.sigma, self.nper_per_year, codes)
        time_handle = random_time(
            self.start, self.num)
        self.set_callback(datetime=time_handle, sec_code=code_handle,
                          open=open_handle, close=close_handle)

    def forward(self) -> None:
        if not self.cur == 0:
            close_value: list[float] = self._callbacks['close'](True)
            self._callbacks['open'](None, close_value, True)


class QAStrategy:
    SIMULATE_SINGLE = 0
    REAL_SINGLE = 1
    SIMULATE_MULTI = 2
    REAL_MULTI = 3


def random_stock_returns(term: int, rate: float, sigma: float, nper_per_year: int) -> list[float]:
    simulated_returns: list[float] = []
    dt = 1/nper_per_year
    for _ in range(1, int(term*nper_per_year)):
        z = np.random.normal()
        simulated_return: float = (rate-(sigma**2/2))*dt + z*sigma*(dt**(1/2))
        simulated_returns.append(simulated_return)
    return simulated_returns


def random_stock_values(price: float, term: int, rate: float, sigma: float, nper_per_year: int) -> list[float]:
    rate = np.array(random_stock_returns(term, rate, sigma, nper_per_year))
    stock_price = [price]
    for i in range(1, int(term*nper_per_year)):
        #  S(i) = S(i-1) * e^(r(i)-1)
        values: float = stock_price[i-1]*math.e**(rate[i-1])
        stock_price.append(values)
    return stock_price


class WsClient:
    def __init__(self, url: str) -> None:
        self.ws = websocket.WebSocket()
        self.ws.connect(url)
        self.stop = False

    def send(self, data: str) -> None:
        self.ws.send(data)

    def recv(self):
        while True:
            if self.stop:
                break
            action = self.ws.recv()
            if action == '[ACK]':
                pass

    def close(self):
        time.sleep(5)
        self.ws.send('[CLOSE]')
        self.stop = True
        self.ws.close()


wscli = WsClient(args.url)


def wrap_klinedata(data: str) -> None:
    wrap('[KLINE]', data)


def wrap_profits(data: str) -> None:
    wrap('[PROFIT]', data)


def wrap_cashes(data: str) -> None:
    wrap('[CASH]', data)


def wrap(tag: str, data: str) -> None:
    wscli.send(tag + data)


def export(task: any) -> None:
    threading.Thread(target=wscli.recv).start()
    task()
    wscli.close()
