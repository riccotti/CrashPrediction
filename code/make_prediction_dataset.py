import sys
import gzip
import json

import pandas as pd

from os import listdir
from collections import defaultdict


def main():

    area = sys.argv[1]      # 'rome' 'tuscany' 'london'

    path = '/media/riccardo/data1/TrackAndKnow/CrashPrediction/'
    path_traintest = path + 'traintest/'

    filenames = defaultdict(list)
    for filename in listdir(path_traintest):
        if area in filename and 'json.gz' in filename:
            index = filename[filename.find('.json.gz') - 1]
            filenames[index].append(filename)

    for index, fn in filenames.items():
        trainset = list()
        testset = list()
        print(index)
        for filename in fn:
            fout = gzip.GzipFile(path_traintest + filename, 'r')
            for row in fout:
                customer_obj = json.loads(row)
                train = customer_obj['train']
                test = customer_obj['test']
                trainset.append(train)
                testset.append(test)
            fout.close()
        df_train = pd.DataFrame(data=trainset)
        df_train['crash'] = df_train['crash'].astype(int)
        # df_train.set_index('uid', inplace=True)
        df_test = pd.DataFrame(data=testset)
        df_test['crash'] = df_test['crash'].astype(int)
        # df_test.set_index('uid', inplace=True)
        print(df_train.head())

        df_train.to_csv(path_traintest + '%s_train_%s.csv.gz' % (area, index), index=False, compression='gzip')
        df_test.to_csv(path_traintest + '%s_test_%s.csv.gz' % (area, index), index=False, compression='gzip')


if __name__ == "__main__":
    main()
