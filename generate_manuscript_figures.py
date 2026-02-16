"""
Created on Thu Aug 1 2024

@author: Coren Lahav
"""


import warnings
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from iqr_outlier_detection import detect_outliers_iqr

warnings.simplefilter(action='ignore', category=FutureWarning)

def percentile(n):
    def percentile_(x):
        return x.quantile(n)
    percentile_.__name__ = 'percentile_{:02.0f}'.format(n*100)
    return percentile_


def pval_text(pval):
    '''pval_text(pval)
    Returns text for p-value in two formats 1) p < 0.01
                                            2) **
    
    Parameters
    ----------
        pval : float
            p-value
    
    Returns
    -------
        tuple with two strings
    '''
    
    if np.isnan(pval):
        return 'NaN', 'NaN'
    elif pval >= 0.05:
        return '= {:.2f}'.format(pval), 'ns'
    elif pval >= 0.01:
        return '= {:.2f}'.format(pval), '*'
    elif pval >= 0.001:
        return '= {:.3f}'.format(pval), '**'
    elif pval >= 0.0001:
        return '< 0.001', '***'
    elif pval < 0.0001:
        return '< 0.0001', '****'
    else:
        return 'Invalid value', 'Invalid value'
    
    
colors = [[0, '#ABC8EE'],
           [0.15, '#3D556F'],
           [0.3, 'orange'],
           [1, 'red']] # Dark blue
white_blue_red = matplotlib.colors.LinearSegmentedColormap.from_list('WhiteBlueRed', colors)

plt.rcParams.update({'font.size': 20})

#%% Load Supplementary Table S8: Serum-plasma correlation
s1 = pd.read_excel(r'data\Supplementary Table S8.xlsx', index_col='SeqId')

# % significant correlations
pct_significant = (s1['Spearman p-value'] < 0.05).mean() * 100
print(f'For {pct_significant:.1f}% of the measured proteins, serum and plasma protein levels were significantly correlated')

# Plot Figure 3C: Association between median plasma-to-serum ratio and plasma-serum protein correlation.
CORR_THR = 0.6
[lower_outlier_threshold, upper_outlier_threshold] = detect_outliers_iqr(np.log2(s1['Plasma to serum ratio']))
s1 = s1.sort_values(by='Plasma to serum ratio', ascending=False)
# Dynamically assign colors: first N are red, rest are blue (example: split at 11)
num_red = 11; num_blue = 4
cmap = matplotlib.colors.ListedColormap(['red'] * num_red + ['blue'] * num_blue)
qtls = s1.loc[s1['Spearman'] >= CORR_THR, 'Plasma to serum ratio'].quantile([0.01,0.99])
fig = plt.figure(figsize=(12,5))
plt.scatter(x=s1['Rank'], y=s1['Plasma to serum ratio'].values, c=s1['Spearman'], vmin=-0.5, vmax=1, cmap=cmap)
ax=plt.gca()
ax.axhline(y=2**upper_outlier_threshold, ls='--', color='gray')
ax.axhline(y=2**lower_outlier_threshold, ls='--', color='gray')
ax.set_yscale('log')
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
space = ' ' * 6
plt.xlabel(u'\u2190 High in plasma' + space + 'Protein rank' + space + u'High in serum \u2192')
plt.ylabel('Plasma to serum ratio')
cax = plt.colorbar()
cax.ax.set_ylabel('Plasma-serum Spearman corr')


# Plot Supplementary Figure S3B: Correlations between plasma and serum proteomes.
jg = sns.jointplot(data=s1, x='Pearson', y='Spearman', marginal_ticks=True)
jg.ax_joint.plot([-0.4,1.02],[-0.4,1.02], c='k', ls='--')
jg.ax_marg_y.tick_params(axis='x', rotation=270)

# Define proteins to annotate with their label alignment
PROTEINS_TO_ANNOTATE = {
    '2796-62': {'ha': 'right', 'va': 'bottom'},
    '5060-62': {'ha': 'right', 'va': 'bottom'},
    '5692-79': {'ha': 'right', 'va': 'bottom'},
    '4673-13': {'ha': 'right', 'va': 'bottom'},
    '4336-2': {'ha': 'right', 'va': 'bottom'},
    '16926-44': {'ha': 'left', 'va': 'top'},
    '9838-4': {'ha': 'left', 'va': 'top'},
    '13955-33': {'ha': 'left', 'va': 'top'}
}

# Annotate specific proteins
for prot_id, alignment in PROTEINS_TO_ANNOTATE.items():
    if prot_id not in s1.index:
        print(f"Warning: Protein {prot_id} not found in data")
        continue
    
    x_val = s1.loc[prot_id, 'Pearson']
    y_val = s1.loc[prot_id, 'Spearman']
    label = s1.loc[prot_id, 'Target Name']
    
    # Plot marker
    jg.ax_joint.plot(x_val, y_val, marker='x', color='r', markersize=6)
    
    # Add label
    jg.ax_joint.text(x_val, y_val, label,
        horizontalalignment=alignment['ha'],
        verticalalignment=alignment['va'],
        fontsize=15
    )
plt.show()


# Plot Supplementary Figure S4: Relationship between Pearson-Spearman correlation differences and protein measurability.
plt.figure(figsize=(7,5))
s1['95th percentile - LoDB'] = s1['95th percentile (log2(RFU))'] - np.log2(s1['LoDB (RFU)'])
s1['Pearson-Spearman'] = s1['Pearson'] - s1['Spearman']
sprmn = spearmanr(s1['95th percentile - LoDB'], s1['Pearson-Spearman'])
ax = sns.histplot(s1, x='95th percentile - LoDB', y='Pearson-Spearman', cbar=True, cbar_kws={'label': 'Number of proteins'})
plt.title(f'Spearman r = {sprmn.statistic:.2f}, p-value {pval_text(sprmn.pvalue)[0]}')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()

#%% Load Supplementary Table S11: Medians and difference between internal and external cohorts
s2 = pd.read_excel(r'data\Supplementary Table S11.xlsx', index_col='SeqId')

# Plot Supplementary Figure S8B: Concordance between internal and external plasma datasets
# Plot Supplementary Figure S9B: Concordance between internal and external serum datasets. 
bins = np.linspace(-5.084463885736609, 4.774182194065542, 101)
CONCORD_ORDER = np.array([['CohortA Plasma, Diff', 'CohortB Plasma, Diff', 'CohortC Plasma, Diff'], ['CohortA Serum, Diff', 'CohortB Serum, Diff', 'CohortC Serum, Diff']])
fig, axes = plt.subplots(nrows=CONCORD_ORDER.shape[0], ncols=CONCORD_ORDER.shape[1]*2, sharey=True, figsize=(15*0.8,20*0.8),  width_ratios=[2,1]*CONCORD_ORDER.shape[1])
for j, i in np.ndindex(CONCORD_ORDER.shape):
    group = CONCORD_ORDER[j, i]
    indication = group.split('\n')[0]
    ax = axes[j, i*2]
    sns.histplot(data=s2, y=group, bins=bins, ax=ax, color='C'+str(i))
    ax.set_title(group.replace(' plasma', '\nplasma').replace(' serum', '\nserum'))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel('Protein count')
    if i == 0:
        ax.set_ylabel("median(paired) - median(external)")
    
    ax = axes[j, i*2+1]
    sns.boxplot(data=s2, y=group, ax=ax, color='C'+str(i), showfliers=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticks([])
    ax.set_ylim([-1, 2.7])
plt.tight_layout()


#%% Load Supplementary Table S14: Scaling factors by cohort
s3 = pd.read_excel(r'data\Supplementary Table S14.xlsx', index_col='SeqId')


# Plot Supplementary Figure S6B: Serum-plasma Spearman correlation distributions of protein measurements in different protein sets.
FILTER_ORDER_DISPLAY = ['Pre-analytical\nexclusions', 'Robust\ncandidates', 'Clinically-relevant\nmarkers']
FILTER_ORDER = [o.replace('\n', ' ') for o in FILTER_ORDER_DISPLAY]
group_protein_count = [(s3['Filter'] == o.replace('\n', ' ')).sum() for o in FILTER_ORDER]
plt.figure(figsize=(9,6))
ax = sns.violinplot(s3, x='Filter', y='Spearman CohortA', hue='Filter', order=FILTER_ORDER,
               hue_order=FILTER_ORDER, palette=['C3', 'C1', 'C2'])
ax.set_title('Modeling pipeline preferentially retained\nserum-plasma agreeing proteins')
plt.xticks([0,1,2], [FILTER_ORDER_DISPLAY[tk] + '\n(n=' + str(group_protein_count[tk]) + ')' for tk in range(len(FILTER_ORDER))])
ax.set_xlabel('Protein group')
ax.set_ylabel("Serum-plasma\nSpearman's r")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


# Plot Supplementary Figure S10: Comparison of linear scaling parameters derived from different cohorts.
PARAM_LIST = ['Spearman', 'Slope', 'Intercept']
plt.rcParams.update({'font.size': 20})
FILTER_LIST = [['Pre-analytical exclusions', 'Robust candidates', 'Clinically-relevant markers'], ['Clinically-relevant markers']]
fig, axes = plt.subplots(nrows=len(FILTER_LIST), ncols=len(PARAM_LIST), figsize=(20,10))
def plot_violin_with_line(ax, data, param):
    """Helper to plot violinplot and add optional horizontal line."""
    if param == 'Slope':
        ax.axhline(y=1, c='gray', ls='--', lw=3)
    elif param == 'Intercept':
        ax.axhline(y=0, c='gray', ls='--', lw=3)
    sns.violinplot(data, color='C0', ax=ax)

for p_idx, param in enumerate(PARAM_LIST):  
    prefix = param + ' '
    cohorts = np.array(['CohortA', 'CohortB', 'CohortC'])
    cols = [prefix + c for c in cohorts]
    
    plot_df = s3[cols].copy()
    plot_df.rename({col: col[len(prefix):] for col in plot_df.columns}, axis=1, inplace=True)
    
    # Box plot
    for f_idx, filtr in enumerate(FILTER_LIST):
        filt = s3['Filter'].isin(filtr)
        ax = axes[f_idx, p_idx]
        plot_violin_with_line(ax, plot_df.loc[filt], param)
        ax.set_title(f'{param} (n={filt.sum()})')
        ax.set_ylabel(f'Serum-plasma\n{param}')
    
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.show()

# Plot Figure 5: Inter-cohort agreement of scaling parameters.
# Plot Supplementary Figure S11: Inter-cohort agreement of serum-plasma protein correlations. 
filtr = FILTER_LIST[0]
for param in ['Spearman', 'Slope', 'Intercept']:
    filt = s3['Filter'].isin(filtr)
    prefix = param + ' '
    cohorts = np.array(['CohortA', 'CohortB', 'CohortC'])
    cols = [prefix + c for c in cohorts]
    
    plot_df = s3[cols].copy()
    plot_df.rename({col: col[len(prefix):] for col in plot_df.columns}, axis=1, inplace=True)
    
    # Pair plot
    rng = [plot_df.loc[filt].min().min(), plot_df.loc[filt].max().max()]
    ax_lim = [rng[0] - (rng[1] - rng[0]) * 0.04, rng[1] + (rng[1] - rng[0]) * 0.04]
    grd = sns.pairplot(plot_df.loc[filt], kind='hist')
    grd.fig.set_size_inches(12,12)
    
    # Define colorbar range for pairplot
    density_lim = {'min': [], 'max': []}
    for r in range(grd.axes.shape[0]):
        for c in range(grd.axes.shape[1]):
            if r != c and grd.axes[r,c] is not None:
                grd.axes[r,c].get_children()[0].set_cmap(white_blue_red)
                cmap = grd.axes[r,c].get_children()[0].get_cmap()
                cl = grd.axes[r,c].get_children()[0].get_clim()
                density_lim['min'].append(cl[0])
                density_lim['max'].append(cl[1])
    
    cbar_min = min(density_lim['min'])
    cbar_max = max(density_lim['max'])
    for r in range(grd.axes.shape[0]):
        for c in range(grd.axes.shape[1]):
            if r != c and grd.axes[r,c] is not None:
                grd.axes[r,c].get_children()[0].set_clim([cbar_min, cbar_max])
    norm = plt.Normalize(cbar_min, cbar_max)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    # Add gray diagonal lines at x=y
    for r in range(grd.axes.shape[0]):
        for c in range(grd.axes.shape[1]):
            ax = grd.axes[r,c]
            if r == c:
                if param == 'Slope':
                    ax.axvline(x=1, c='gray', ls='--', lw=3)
                if param == 'Intercept':
                    ax.axvline(x=0, c='gray', ls='--', lw=3)
            if r != c and grd.axes[r,c] is not None:
                ax.set_xlim(ax_lim)
                ax.set_ylim(ax_lim)
                if param == 'Slope':
                    ax.set_xticks([0,1,2])
                    ax.set_yticks([0,1,2])
                    ax.set_xlim([-1,3])
                    ax.set_ylim([-1,3])
                if param == 'Intercept':
                    ax.set_xlim([-23,23])
                    ax.set_ylim([-23,23])
                ax.plot(rng, rng, ls=':', c='k', lw=3)
                prsn = pearsonr(plot_df.loc[filt, plot_df.columns[r]], plot_df.loc[filt, plot_df.columns[c]])
                ax.set_title(f'r = {prsn.statistic:.2f}')
    plt.suptitle(f'{param} (n={len(filt)})', fontsize=40)
    grd.fig.tight_layout()
    
    # Add the colorbar
    cbar=grd.fig.colorbar(sm, ax=grd.axes)
    for child_ax in grd.fig.get_children():
        if type(child_ax) == matplotlib.axes._axes.Axes:
            child_ax.set_ylabel('Protein count')
    plt.show()