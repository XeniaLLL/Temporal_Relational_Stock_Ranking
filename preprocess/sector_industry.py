import argparse
from datetime import datetime
import copy
import json
import numpy as np
import operator
import os
import pandas as pd

class SectorPreprocessor:
    def __init__(self, data_path, market_name):
        self.data_path = data_path
        self.date_format = '%Y-%m-%d %H:%M:%S'
        self.market_name = market_name

    def generate_sector_relation(self, industry_ticker_file,
                                 selected_tickers_fname):
        selected_tickers = np.genfromtxt(
            os.path.join(self.data_path, '..', selected_tickers_fname),
            dtype=str, delimiter='\t', skip_header=False
        ) # 从数据path得到..退出到新的一层得到需要的股票信息; np.genfromtxt 能支持多种filename 形式,甚至是gzip和bz2两种压缩的形式
        print('#tickers selected:', len(selected_tickers))
        ticker_index = {}
        for index, ticker in enumerate(selected_tickers):
            ticker_index[ticker] = index # ticker id编号
        with open(industry_ticker_file, 'r') as fin: # industry 信息--> predefined
            industry_tickers = json.load(fin)
        print('#industries: ', len(industry_tickers))
        valid_industry_count = 0
        valid_industry_index = {}
        for industry in industry_tickers.keys(): # 给tiker 不只一个的industry 进行编号
            if len(industry_tickers[industry]) > 1: # 过滤只有一个股票的产业--> 因为这样没有关系信息
                valid_industry_index[industry] = valid_industry_count
                valid_industry_count += 1
        one_hot_industry_embedding = np.identity(valid_industry_count + 1,
                                                 dtype=int) # todo 为什么有多一个embedding
        ticker_relation_embedding = np.zeros(
            [len(selected_tickers), len(selected_tickers),
             valid_industry_count + 1], dtype=int) #　3D Tensor, n_tikers* n_tikers 然后有n_valid_industry 个这样的图的邻接矩阵
        print(ticker_relation_embedding[0][0].shape)
        for industry in valid_industry_index.keys():
            cur_ind_tickers = industry_tickers[industry]
            if len(cur_ind_tickers) <= 1:
                print('shit industry:', industry)
                continue
            ind_ind = valid_industry_index[industry]
            for i in range(len(cur_ind_tickers)):
                # tiker自己的就是设置为对角位置的embedding 但是多了的是最后一维用1标记了
                left_tic_ind = ticker_index[cur_ind_tickers[i]]
                ticker_relation_embedding[left_tic_ind][left_tic_ind] = \
                    copy.copy(one_hot_industry_embedding[ind_ind])
                ticker_relation_embedding[left_tic_ind][left_tic_ind][-1] = 1 # 最后一维访问的是idx=96 的位置

                for j in range(i + 1, len(cur_ind_tickers)): # 这样只用处理一个三角阵
                    right_tic_ind = ticker_index[cur_ind_tickers[j]]
                    ticker_relation_embedding[left_tic_ind][right_tic_ind] = \
                        copy.copy(one_hot_industry_embedding[ind_ind])
                    ticker_relation_embedding[right_tic_ind][left_tic_ind] = \
                        copy.copy(one_hot_industry_embedding[ind_ind])
                    # print(right_tic_ind)

        # handle shit industry and n/a tickers
        for i in range(len(selected_tickers)):
            ticker_relation_embedding[i][i][-1] = 1 # 和对角阵一样说明是单个relation ???
        print(ticker_relation_embedding.shape)
        np.save(self.market_name + '_industry_relation',
                ticker_relation_embedding)


if __name__ == '__main__':
    desc = "pre-process sector data market by market, including listing all " \
           "trading days, all satisfied stocks (5 years & high price), " \
           "normalizing and compansating data"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-path', help='path of EOD data')
    parser.add_argument('-market', help='market name')
    args = parser.parse_args()

    if args.path is None:
        args.path = '../data/google_finance'
    if args.market is None:
        args.market = 'NASDAQ'

    processor = SectorPreprocessor(args.path, args.market)

    processor.generate_sector_relation(
        os.path.join('../data/relation/sector_industry/',
                     processor.market_name + '_industry_ticker.json'),
        processor.market_name + '_tickers_qualify_dr-0.98_min-5_smooth.csv'
    )