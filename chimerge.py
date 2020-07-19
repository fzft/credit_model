import copy
import pandas as pd
import scipy.stats as ss
import numpy as np
import  matplotlib.pyplot as plt

# 违约分 30 12     国籍分 19 23    p(不违约) = 30 / 42 = 0.7143  p(违约) = 1- p(不违约) = 0.2857
# p(意大利) = 19 / (19+23) = 19 / 42 = 0.45238   p(南斯拉夫) = 1-p（意大利） = 0.5476
# 假设独立 expect
# p(不违约&意大利) = 0.45238 * 0.7143 = 0.323135034
# p(不违约&南斯拉夫) = 0.5476 * 0.7143 = 0.39115
# p(违约&南斯拉夫) = 0.2857 * 0.5476 = 0.15644932
# p(违约&意大利) = 0.2857 * 0.45238 = 0.129244966
# expect(不违约&意大利) = p(不违约&意大利) * total = 0.323135034 * 42 = 13.571671428
# expect(不违约&南斯拉夫) = p(不违约&南斯拉夫) * total = 0.39115 * 42 = 16.4283
# expect(违约&南斯拉夫) = p(违约&南斯拉夫) * total = 0.15644932 * 42 = 6.57087144
# expect(违约&意大利) = p(违约&意大利) * total = 0.129244966 * 42 = 5.428288572
# chi2-square = (14 - 13.5723)**2 / 13.5723 + (16 -16.4283 )**2 / 16.4283 + ( 5 -5.428288572)**2 / 5.428288572 + (7-6.57087144)**2/6.57087144 =0.08646124271066821

def M_chiMerge(dfname, colname, target, method='MaxInt', maxint=5, minchi2=3.84, vartype='N', outtype=1):
    dfana = copy.copy(dfname[[colname, target]])
    if dfana[colname].isna().sum() > 0:
        dfana = dfana[dfana[colname].notna()]
    n_levels = dfana[colname].nunique()
    if method == 'MaxInt' and n_levels <= maxint:
        return []
    if n_levels > 100:
        pass
    dftab1 = pd.crosstab(dfana[colname], dfana[target])
    dftab1.reset_index(inplace=True)
    dftab1['badrate'] = dftab1[1] / (dftab1[0] + dftab1[1])
    if vartype == 'C':
        dftab1.sort_values('badrate', inplace=True)
        dftab1.reset_index(drop=True, inplace=True)
    dfres = pd.DataFrame({colname: dftab1[colname], 'trans': dftab1[colname]})
    normalbinindex = dftab1.query("badrate > 0 and badrate < 1").index
    if normalbinindex.size < n_levels:
        initpos = normalbinindex[0]
        if initpos > 0:
            dfres.loc[0: initpos, 'trans'] = dfres.loc[initpos, 'trans']
        for i in range(initpos, n_levels):
            if dftab1.loc[i, 'badrate'] == 0 or dftab1.loc[i, 'badrate'] == 1:
                dfres.loc[i, 'trans'] = dfres.loc[i - 1, 'trans']
        dftab1[colname] = dfres.trans
        dftab1 = dftab1.groupby(colname, as_index=False).agg('sum')
        n_levels = dftab1[colname].nunique()

        if vartype == 'C':
            dftab1['badrate'] = dftab1[1] / (dftab1[0] + dftab1[1])
            dftab1.sort_values('badrate', inplace=True)
            dftab1.reset_index(drop=True, inplace=True)
    for i in range(n_levels - 1):
        dftab1.loc[i, 'chi2'] = ss.chi2_contingency(dftab1.loc[i: i + 1, [0, 1]])[0]
    #
    while True:
        minindex = dftab1.index[dftab1.chi2 == min(dftab1.chi2)][0]
        mincat = dftab1.loc[minindex, colname]

        if method == 'MaxInt':
            if dftab1.shape[0] <= maxint:
                break

        else:
            if dftab1.loc[minindex, 'chi2'] >= minchi2:
                break
        newcat = dftab1.loc[minindex + 1, colname]
        dfres.loc[dfres['trans'] == mincat, 'trans'] = newcat
        dftab1.loc[minindex, colname] = newcat

        dftab1 = dftab1.groupby(colname, as_index=False, observed=True).agg('sum')

        if vartype == 'C':
            dftab1['badrate'] = dftab1[1] / (dftab1[0] + dftab1[1])
            dftab1.sort_values('badrate', inplace=True)
            dftab1.reset_index(drop=True, inplace=True)
        dftab1.loc[dftab1.index[-1], 'chi2'] = np.NaN
        if minindex == dftab1.shape[0] - 1:
            minindex -= 1
        dftab1.loc[minindex, 'chi2'] = ss.chi2_contingency(dftab1.loc[minindex: minindex + 1, [0, 1]])[0]

    dfres.trans = dfres.trans.astype('str')
    dfres.columns = [colname, colname + '_bin']

    if vartype == 'N':
        dfresg = dfres.groupby(colname + '_bin')
        dfout = dfresg.min()
        dfout.columns = ['[']
        dfout[']'] = dfresg.max()
        print(dfout)
    elif vartype == 'C':
        dftmp = copy.copy(dfres)
        dftmp[colname] = dftmp[colname].astype('str') + ''
        print(dftmp.groupby(colname + '_bin').sum())
    if outtype == 1:
        return dfres.set_index(colname)
    elif outtype == 2:
        tmp = dfname[[colname]]
        tmp2 = pd.merge(dfname[[colname]], dfres, how='left', on=colname)[colname + '_bin']
        return tmp2


def M_CalcWOE(colname, target, output=True):
    dftbl = pd.crosstab(colname.fillna('NA'), target, normalize='columns')
    dftbl.columns = ['goodpct', 'badpct']
    dftbl['WOE'] = np.log(dftbl.goodpct / dftbl.badpct) * 100
    IV = sum((dftbl.goodpct - dftbl.badpct) * dftbl.WOE) / 100
    if output:
        print('IV : %.3f' % IV)
        dftbl['WOE'].plot.bar()
        plt.gca().set_xticklabels(dftbl.index, rotation='vertical')
    return {'WOE': dftbl['WOE'], 'IV': IV}


if __name__ == '__main__':
    dfraw = pd.read_csv('BankData/ScoreCard.csv', encoding='GBK')
    dfres = M_chiMerge(dfraw, '年龄', '是否违约', outtype=2)
    M_CalcWOE(dfres, dfraw.是否违约)
