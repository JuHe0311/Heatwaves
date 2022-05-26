### Imports ###
import pandas as pd
import argparse

### Argparser ###
def make_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', "--data", help="Give the path to the cluster gv dataset.",
                       type=str)

    return parser

parser = make_argparser()
args = parser.parse_args()
gv = pd.read_csv(args.data)
gv['time']=pd.to_datetime(gv['time'])
gv['month'] = gv.time.dt.month

# calculate seasons of the heat wave family as the 10th (start of season) and 90th qantile (end of season) of the month distribution of the nodes
season = [gv.month.quantile(q=0.1),gv.month.quantile(q=0.9)]

print(season)

