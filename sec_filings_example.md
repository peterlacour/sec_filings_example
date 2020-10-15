# 10-Q SEC Filing Sentiment and Similiarity Metrics Example Notebook

...


```python
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import multiprocessing as mp
import matplotlib.pyplot as plt
from ast import literal_eval
from tqdm.notebook import tqdm
from finpie import historical_prices
from nltk.corpus import stopwords as nltk_stopwords
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# set plot styles
plt.style.use('seaborn')
plt.rcParams['text.color'] = 'black'
plt.rcParams[ "figure.figsize"] = (20, 10)

# load custom classes
from classes.sec_class import SecData
from classes.wordcloud_class import wordCloud
from classes.text_processing_class import textPreProcess

sec = SecData()
wc = wordCloud()
tpp = textPreProcess()

```

### To do:
- add 10-Ks to fill the gap
- cleaning:
    - clean docs of 'Table of Contents'
    - clean docs of Management’s Discussion and Analysis of Financial Condition and Results of Operations 
    - clean docs of forward looking statement or other common texts
    - clean docs of table description such as: ''the following table shows'
    - clean docs of table footnotes '(1)'
    - clean other things and improve regex..

## Download 10-Q SEC Filings


```python
cik_dict = { 'AAPL': '0000320193' } # 'XOM': '0000034088', 'TSLA': '0001318605', 'JNJ': '0000200406' }
ten_qs = {}
for ticker, cik in tqdm(cik_dict.items()):
    ten_qs[ticker] = sec.get_10qs( cik )
```


    HBox(children=(FloatProgress(value=0.0, max=1.0), HTML(value='')))


    



```python
ticker_dict = {}
for ticker in cik_dict.keys():
    print(ticker)
    mdna = {}
    for file_date, doc in tqdm(ten_qs[ticker].items()):
        mdna[file_date] = sec.get_mdna(doc)
    df = pd.DataFrame(mdna, index = ['MDnA']).transpose()
    ticker_dict[ticker] = df
    #display(df.head())
# example
print(ticker, ' example')
display(df.head())
print(df.iloc[0].values[0][:1000])
```

    AAPL



    HBox(children=(FloatProgress(value=0.0, max=80.0), HTML(value='')))


    
    AAPL  example



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MDnA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-07-31</th>
      <td>. Management’s Discussion and Analysis of Fina...</td>
    </tr>
    <tr>
      <th>2020-05-01</th>
      <td>.Management’s Discussion and Analysis of Finan...</td>
    </tr>
    <tr>
      <th>2020-01-29</th>
      <td>.Management’s Discussion and Analysis of Finan...</td>
    </tr>
    <tr>
      <th>2019-07-31</th>
      <td>.Management’s Discussion and Analysis of Finan...</td>
    </tr>
    <tr>
      <th>2019-05-01</th>
      <td>.Management’s Discussion and Analysis of Finan...</td>
    </tr>
  </tbody>
</table>
</div>


    . Management’s Discussion and Analysis of Financial Condition and Results of OperationsThis section and other parts of this Quarterly Report on Form 10-Q (“Form 10-Q”) contain forward-looking statements, within the meaning of the Private Securities Litigation Reform Act of 1995, that involve risks and uncertainties. Forward-looking statements provide current expectations of future events based on certain assumptions and include any statement that does not directly relate to any historical or current fact. For example, statements in this Form 10-Q regarding the potential future impact of the COVID-19 pandemic on the Company’s business and results of operations are forward-looking statements. Forward-looking statements can also be identified by words such as “future,” “anticipates,” “believes,” “estimates,” “expects,” “intends,” “plans,” “predicts,” “will,” “would,” “could,” “can,” “may,” and similar terms. Forward-looking statements are not guarantees of future performance and the Compa



```python
# Illustrative Example:
ticker = 'AAPL'
df = ticker_dict[ticker]
```

## Word Clouds


```python
# create classes
wc = wordCloud()
tpp = textPreProcess()
tpp.textColumn = 'MDnA'

# load stopwords
stopword_files = glob.glob('./stopwords/*.txt')
stopwords = nltk_stopwords.words('english')
for file in stopword_files:
    stopwords += list(pd.read_csv(file).iloc[:,0])
stopwords = [ word.lower() for word in stopwords if type(word) == type("") ]
tpp.stopwords = stopwords



```


```python
# split dataframe
dflist = np.array_split(df,  mp.cpu_count() )
with mp.Pool(mp.cpu_count()) as pool:
    dfs = list(tqdm(pool.imap( tpp.lemmatize,  [ d for d in dflist ] ), total = len(dflist) ))
df = pd.concat( dfs )

wordCloudDict = { f'{ticker}: 10-Q Management\'s Discussion and Analysis': df.lemmatised_text }
wc.create_word_cloud( wordCloudDict,  masks = f'./logos/{ticker.lower()}_logo', title = "company", columns = 1, rows = 1 )

```


    HBox(children=(FloatProgress(value=0.0, max=4.0), HTML(value='')))


    


    /Users/PeterlaCour/Documents/Research.nosync/SEC/sec_github/classes/wordcloud_class.py:70: MatplotlibDeprecationWarning: Adding an axes using the same arguments as a previous axes currently reuses the earlier instance.  In a future version, a new instance will always be created and returned.  Meanwhile, this warning can be suppressed, and the future behavior ensured, by passing a unique label to each axes instance.
      ax.append( fig.add_subplot(rows, columns, count+1 ) )



![png](output_10_3.png)



```python

```

...

## Lexica based sentiment example using Loughran McDonald Dictionary


```python
# load dictionary
lmcd_dictionary = pd.read_csv('LoughranMcDonald_MasterDictionary_2018.csv')
columns = ['Word', 'Negative', 'Positive', 'Uncertainty', 'Litigious', 'Constraining', 'Superfluous', 'Interesting']
lmcd_dictionary = lmcd_dictionary[columns]
lmcd_dictionary.dropna(subset = ['Word'], inplace = True, axis = 0)
lmcd_dictionary.reset_index(drop = True, inplace = True)

def lmcd_sentiment(df_idx):
    df_idx = df_idx.copy()
    text = df_idx.text.values[0] #' '.join( literal_eval( df_idx.lemmatised_text.values[0] ) )
    for i, word in enumerate(lmcd_dictionary.Word):
        if word in text.upper():
            df_idx[columns[1:]] += lmcd_dictionary.loc[i, columns[1:]].values
    return pd.DataFrame(df_idx)
```


```python
for col in columns[1:]:
    df[col] = 0
# concatenated texts...
df['text'] = [ ' '.join( literal_eval( txt ) ) for txt in df.lemmatised_text ]

# calculate lexica based sentiment
with mp.Pool(mp.cpu_count()) as pool:
    dfs = list(tqdm(pool.imap( lmcd_sentiment, [d for d in np.array_split(df, len(df))] ), total = len(df) ))
df = pd.concat( dfs )
df.index = pd.to_datetime(df.index)
df.sort_index(inplace = True)
display(df.head())
```


    HBox(children=(FloatProgress(value=0.0, max=80.0), HTML(value='')))


    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MDnA</th>
      <th>lemmatised_text</th>
      <th>Negative</th>
      <th>Positive</th>
      <th>Uncertainty</th>
      <th>Litigious</th>
      <th>Constraining</th>
      <th>Superfluous</th>
      <th>Interesting</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1994-01-26</th>
      <td>.  Management's Discussion and Analysis of Fin...</td>
      <td>['management', 'discussion', 'analysis', 'fina...</td>
      <td>56252</td>
      <td>46207</td>
      <td>26117</td>
      <td>18085</td>
      <td>14067</td>
      <td>0</td>
      <td>6027</td>
      <td>management discussion analysis financial condi...</td>
    </tr>
    <tr>
      <th>1994-08-12</th>
      <td>.  Management's Discussion and Analysis of Fin...</td>
      <td>['management', 'discussion', 'analysis', 'fina...</td>
      <td>88396</td>
      <td>48216</td>
      <td>36164</td>
      <td>20094</td>
      <td>18085</td>
      <td>0</td>
      <td>6027</td>
      <td>management discussion analysis financial condi...</td>
    </tr>
    <tr>
      <th>1995-02-09</th>
      <td>.  Management's Discussion and Analysis of Fin...</td>
      <td>['management', 'discussion', 'analysis', 'fina...</td>
      <td>86387</td>
      <td>60270</td>
      <td>36162</td>
      <td>30139</td>
      <td>18085</td>
      <td>0</td>
      <td>4018</td>
      <td>management discussion analysis financial condi...</td>
    </tr>
    <tr>
      <th>1995-05-15</th>
      <td>.  Management's Discussion and Analysis of Fin...</td>
      <td>['management', 'discussion', 'analysis', 'fina...</td>
      <td>94423</td>
      <td>58261</td>
      <td>38171</td>
      <td>30139</td>
      <td>18085</td>
      <td>0</td>
      <td>4018</td>
      <td>management discussion analysis financial condi...</td>
    </tr>
    <tr>
      <th>1995-08-11</th>
      <td>.  Management's Discussion and Analysis of Fin...</td>
      <td>['management', 'discussion', 'analysis', 'fina...</td>
      <td>90405</td>
      <td>62279</td>
      <td>40180</td>
      <td>30139</td>
      <td>20094</td>
      <td>0</td>
      <td>4018</td>
      <td>management discussion analysis financial condi...</td>
    </tr>
  </tbody>
</table>
</div>



```python

```


```python
# plot absolute 
df.index = pd.to_datetime(df.index)
df.sort_index(inplace = True)
df[columns[1:]].plot()
plt.title(f'Absolute sentiment scores of {ticker}\'s 10-Q filings', fontsize = 24)
plt.show()

# Get price data
prices = historical_prices(ticker)
prices['adj_close'][df.index[0]:].plot()
plt.title(f'{ticker} stock price')
plt.show()
```


![png](output_17_0.png)



![png](output_17_1.png)



```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MDnA</th>
      <th>lemmatised_text</th>
      <th>Negative</th>
      <th>Positive</th>
      <th>Uncertainty</th>
      <th>Litigious</th>
      <th>Constraining</th>
      <th>Superfluous</th>
      <th>Interesting</th>
      <th>text</th>
      <th>quarter_returns</th>
      <th>doc_length</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1994-01-26</th>
      <td>.  Management's Discussion and Analysis of Fin...</td>
      <td>['management', 'discussion', 'analysis', 'fina...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>management discussion analysis financial condi...</td>
      <td>0.045145</td>
      <td>17564</td>
    </tr>
    <tr>
      <th>1994-08-12</th>
      <td>.  Management's Discussion and Analysis of Fin...</td>
      <td>['management', 'discussion', 'analysis', 'fina...</td>
      <td>0.080765</td>
      <td>-0.282338</td>
      <td>-0.047665</td>
      <td>-0.235840</td>
      <td>-0.115794</td>
      <td>NaN</td>
      <td>-0.312241</td>
      <td>management discussion analysis financial condi...</td>
      <td>0.263522</td>
      <td>25538</td>
    </tr>
    <tr>
      <th>1995-02-09</th>
      <td>.  Management's Discussion and Analysis of Fin...</td>
      <td>['management', 'discussion', 'analysis', 'fina...</td>
      <td>-0.014936</td>
      <td>0.259966</td>
      <td>0.007917</td>
      <td>0.511859</td>
      <td>0.007973</td>
      <td>NaN</td>
      <td>-0.328018</td>
      <td>management discussion analysis financial condi...</td>
      <td>0.002749</td>
      <td>25336</td>
    </tr>
    <tr>
      <th>1995-05-15</th>
      <td>.  Management's Discussion and Analysis of Fin...</td>
      <td>['management', 'discussion', 'analysis', 'fina...</td>
      <td>-0.033814</td>
      <td>-0.145507</td>
      <td>-0.066933</td>
      <td>-0.116042</td>
      <td>-0.116042</td>
      <td>NaN</td>
      <td>-0.116042</td>
      <td>management discussion analysis financial condi...</td>
      <td>-0.010162</td>
      <td>28662</td>
    </tr>
    <tr>
      <th>1995-08-11</th>
      <td>.  Management's Discussion and Analysis of Fin...</td>
      <td>['management', 'discussion', 'analysis', 'fina...</td>
      <td>-0.065347</td>
      <td>0.043517</td>
      <td>0.027571</td>
      <td>-0.023807</td>
      <td>0.084635</td>
      <td>NaN</td>
      <td>-0.023807</td>
      <td>management discussion analysis financial condi...</td>
      <td>-0.337220</td>
      <td>29361</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2019-05-01</th>
      <td>.Management’s Discussion and Analysis of Finan...</td>
      <td>['discussion', 'analysis', 'financial', 'condi...</td>
      <td>-0.108699</td>
      <td>0.003282</td>
      <td>0.121315</td>
      <td>0.121315</td>
      <td>-0.055874</td>
      <td>NaN</td>
      <td>-0.439343</td>
      <td>discussion analysis financial condition result...</td>
      <td>0.015867</td>
      <td>22759</td>
    </tr>
    <tr>
      <th>2019-07-31</th>
      <td>.Management’s Discussion and Analysis of Finan...</td>
      <td>['discussion', 'analysis', 'financial', 'condi...</td>
      <td>0.129433</td>
      <td>-0.241217</td>
      <td>-0.124689</td>
      <td>-0.078620</td>
      <td>-0.020866</td>
      <td>NaN</td>
      <td>-0.078620</td>
      <td>discussion analysis financial condition result...</td>
      <td>0.532810</td>
      <td>24701</td>
    </tr>
    <tr>
      <th>2020-01-29</th>
      <td>.Management’s Discussion and Analysis of Finan...</td>
      <td>['discussion', 'analysis', 'financial', 'condi...</td>
      <td>-0.092430</td>
      <td>0.343675</td>
      <td>0.155089</td>
      <td>-0.118171</td>
      <td>0.198488</td>
      <td>NaN</td>
      <td>0.567621</td>
      <td>discussion analysis financial condition result...</td>
      <td>-0.106629</td>
      <td>15757</td>
    </tr>
    <tr>
      <th>2020-05-01</th>
      <td>.Management’s Discussion and Analysis of Finan...</td>
      <td>['discussion', 'analysis', 'financial', 'condi...</td>
      <td>0.197822</td>
      <td>-0.160350</td>
      <td>-0.169577</td>
      <td>-0.224939</td>
      <td>0.013601</td>
      <td>NaN</td>
      <td>-0.224939</td>
      <td>discussion analysis financial condition result...</td>
      <td>0.474351</td>
      <td>20330</td>
    </tr>
    <tr>
      <th>2020-07-31</th>
      <td>. Management’s Discussion and Analysis of Fina...</td>
      <td>['management', 'discussion', 'analysis', 'fina...</td>
      <td>-0.093186</td>
      <td>-0.081936</td>
      <td>0.060874</td>
      <td>0.105065</td>
      <td>0.111688</td>
      <td>NaN</td>
      <td>-0.005430</td>
      <td>management discussion analysis financial condi...</td>
      <td>NaN</td>
      <td>20441</td>
    </tr>
  </tbody>
</table>
<p>80 rows × 12 columns</p>
</div>




```python
# add quarter on quarter returns to dataframe
# should really start at next days opening price but keeping this for simplicity for now
df['quarter_returns'] = prices['adj_close'].loc[df.index].pct_change().shift(-1)

# rescale sentiment based on doc length -> "average sentiments"
df['doc_length'] = [ len(d) for d in df.lemmatised_text ]
for col in columns[1:]:
    df[col] = df[col] / df['doc_length']

# Show standardised sentiment
df.index = pd.to_datetime(df.index)
df.sort_index(inplace = True)
df[columns[1:]].plot()
plt.title(f'Sentiment scores of {ticker}\'s 10-Q filings', fontsize = 24)
plt.show()

# correlation between returns and average metrics
print('Correlation between returns and standardised metrics:')
display(df[columns[1:] + ['quarter_returns']].corr())
#sns.heatmap(df[columns[1:] + ['quarter_returns']].corr(), cmap = 'Blues')
#plt.show()
```


```python
# calculating % change of metric from report to report
# superfluous has many zero values, still need to deal with this issue in percentage calculation,
# and for other companies other metrics might have missing values
df[columns[1:]] = df[columns[1:]].pct_change()

# correlation between percentage changes and returns
display(df[columns[1:] + ['quarter_returns']].corr())
plt.title('Heatmap of correlations between sentiment percentage changes and quarterly returns', fontsize = 24)
sns.heatmap(df[columns[1:] + ['quarter_returns']].corr(), cmap = 'Blues' )
plt.show()
df.to_csv(f'{ticker}.csv')
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Negative</th>
      <th>Positive</th>
      <th>Uncertainty</th>
      <th>Litigious</th>
      <th>Constraining</th>
      <th>Superfluous</th>
      <th>Interesting</th>
      <th>quarter_returns</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Negative</th>
      <td>1.000000</td>
      <td>0.224751</td>
      <td>0.232990</td>
      <td>-0.025736</td>
      <td>0.422873</td>
      <td>NaN</td>
      <td>0.255966</td>
      <td>-0.168920</td>
    </tr>
    <tr>
      <th>Positive</th>
      <td>0.224751</td>
      <td>1.000000</td>
      <td>0.622460</td>
      <td>0.475404</td>
      <td>0.503197</td>
      <td>NaN</td>
      <td>0.316076</td>
      <td>-0.000452</td>
    </tr>
    <tr>
      <th>Uncertainty</th>
      <td>0.232990</td>
      <td>0.622460</td>
      <td>1.000000</td>
      <td>0.516805</td>
      <td>0.550661</td>
      <td>NaN</td>
      <td>0.416318</td>
      <td>0.001407</td>
    </tr>
    <tr>
      <th>Litigious</th>
      <td>-0.025736</td>
      <td>0.475404</td>
      <td>0.516805</td>
      <td>1.000000</td>
      <td>0.362877</td>
      <td>NaN</td>
      <td>0.320827</td>
      <td>0.076563</td>
    </tr>
    <tr>
      <th>Constraining</th>
      <td>0.422873</td>
      <td>0.503197</td>
      <td>0.550661</td>
      <td>0.362877</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>0.198583</td>
      <td>-0.122630</td>
    </tr>
    <tr>
      <th>Superfluous</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Interesting</th>
      <td>0.255966</td>
      <td>0.316076</td>
      <td>0.416318</td>
      <td>0.320827</td>
      <td>0.198583</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>-0.036683</td>
    </tr>
    <tr>
      <th>quarter_returns</th>
      <td>-0.168920</td>
      <td>-0.000452</td>
      <td>0.001407</td>
      <td>0.076563</td>
      <td>-0.122630</td>
      <td>NaN</td>
      <td>-0.036683</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



![png](output_20_1.png)



```python

```

## Similarity Metrics


```python

```

### Cosine Similarity Scores of 10-Q's


```python
# cosine similarity with tfidf or word2vec
tfidf = TfidfVectorizer()
tfidf_df = tfidf.fit_transform(df.text).toarray()
rows = []
for j in range(len(tfidf_df)):
    columns = []
    for i in range(len(tfidf_df)):
        columns.append( cosine_similarity(tfidf_df[j].reshape(1, -1), tfidf_df[i].reshape(1, -1))[0][0] )
    rows.append(columns)

```


```python
cosine_similiarity = pd.DataFrame(rows, columns = df.index, index = df.index )
display(cosine_similiarity)

# similiarity heatmap plot
sns.heatmap(cosine_similiarity, cmap = 'Blues')
plt.title('Cosine Similiarity of 10-Q\'s based on TFIDF', fontsize = 24 )
plt.show()

# plot of similiarity of previous reports to most recent report
plt.plot(cosine_similiarity.iloc[:,-1])
plt.title('Cosine similarity of most recent 10-Q with previous 10-Qs', fontsize = 24 )
plt.show()
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1994-01-26</th>
      <th>1994-08-12</th>
      <th>1995-02-09</th>
      <th>1995-05-15</th>
      <th>1995-08-11</th>
      <th>1996-02-12</th>
      <th>1996-05-13</th>
      <th>1996-08-12</th>
      <th>1997-02-10</th>
      <th>1997-05-12</th>
      <th>...</th>
      <th>2017-08-02</th>
      <th>2018-02-02</th>
      <th>2018-05-02</th>
      <th>2018-08-01</th>
      <th>2019-01-30</th>
      <th>2019-05-01</th>
      <th>2019-07-31</th>
      <th>2020-01-29</th>
      <th>2020-05-01</th>
      <th>2020-07-31</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1994-01-26</th>
      <td>1.000000</td>
      <td>0.878369</td>
      <td>0.858761</td>
      <td>0.845633</td>
      <td>0.848178</td>
      <td>0.851594</td>
      <td>0.836922</td>
      <td>0.829260</td>
      <td>0.810267</td>
      <td>0.816604</td>
      <td>...</td>
      <td>0.595510</td>
      <td>0.589953</td>
      <td>0.592078</td>
      <td>0.586235</td>
      <td>0.628450</td>
      <td>0.600197</td>
      <td>0.601948</td>
      <td>0.569397</td>
      <td>0.544281</td>
      <td>0.558448</td>
    </tr>
    <tr>
      <th>1994-08-12</th>
      <td>0.878369</td>
      <td>1.000000</td>
      <td>0.929547</td>
      <td>0.918915</td>
      <td>0.913510</td>
      <td>0.918111</td>
      <td>0.884871</td>
      <td>0.881480</td>
      <td>0.869446</td>
      <td>0.861637</td>
      <td>...</td>
      <td>0.585610</td>
      <td>0.583967</td>
      <td>0.585925</td>
      <td>0.582726</td>
      <td>0.626036</td>
      <td>0.589286</td>
      <td>0.594640</td>
      <td>0.553068</td>
      <td>0.525156</td>
      <td>0.541229</td>
    </tr>
    <tr>
      <th>1995-02-09</th>
      <td>0.858761</td>
      <td>0.929547</td>
      <td>1.000000</td>
      <td>0.982277</td>
      <td>0.974758</td>
      <td>0.944270</td>
      <td>0.909295</td>
      <td>0.907085</td>
      <td>0.882686</td>
      <td>0.880713</td>
      <td>...</td>
      <td>0.589546</td>
      <td>0.583277</td>
      <td>0.584297</td>
      <td>0.578835</td>
      <td>0.631124</td>
      <td>0.592735</td>
      <td>0.596862</td>
      <td>0.561720</td>
      <td>0.532077</td>
      <td>0.549317</td>
    </tr>
    <tr>
      <th>1995-05-15</th>
      <td>0.845633</td>
      <td>0.918915</td>
      <td>0.982277</td>
      <td>1.000000</td>
      <td>0.986668</td>
      <td>0.941737</td>
      <td>0.908345</td>
      <td>0.906681</td>
      <td>0.881417</td>
      <td>0.882195</td>
      <td>...</td>
      <td>0.603517</td>
      <td>0.590433</td>
      <td>0.594479</td>
      <td>0.589204</td>
      <td>0.641515</td>
      <td>0.610961</td>
      <td>0.615111</td>
      <td>0.579453</td>
      <td>0.547014</td>
      <td>0.569355</td>
    </tr>
    <tr>
      <th>1995-08-11</th>
      <td>0.848178</td>
      <td>0.913510</td>
      <td>0.974758</td>
      <td>0.986668</td>
      <td>1.000000</td>
      <td>0.951487</td>
      <td>0.917808</td>
      <td>0.916056</td>
      <td>0.888504</td>
      <td>0.888525</td>
      <td>...</td>
      <td>0.606029</td>
      <td>0.596179</td>
      <td>0.599901</td>
      <td>0.594410</td>
      <td>0.644509</td>
      <td>0.612213</td>
      <td>0.616117</td>
      <td>0.581286</td>
      <td>0.546911</td>
      <td>0.569013</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2019-05-01</th>
      <td>0.600197</td>
      <td>0.589286</td>
      <td>0.592735</td>
      <td>0.610961</td>
      <td>0.612213</td>
      <td>0.609639</td>
      <td>0.627548</td>
      <td>0.635700</td>
      <td>0.621395</td>
      <td>0.645254</td>
      <td>...</td>
      <td>0.898254</td>
      <td>0.870324</td>
      <td>0.881505</td>
      <td>0.877668</td>
      <td>0.953209</td>
      <td>1.000000</td>
      <td>0.976697</td>
      <td>0.936386</td>
      <td>0.855574</td>
      <td>0.893976</td>
    </tr>
    <tr>
      <th>2019-07-31</th>
      <td>0.601948</td>
      <td>0.594640</td>
      <td>0.596862</td>
      <td>0.615111</td>
      <td>0.616117</td>
      <td>0.617252</td>
      <td>0.635382</td>
      <td>0.641832</td>
      <td>0.627287</td>
      <td>0.648667</td>
      <td>...</td>
      <td>0.882871</td>
      <td>0.850796</td>
      <td>0.860208</td>
      <td>0.856758</td>
      <td>0.973040</td>
      <td>0.976697</td>
      <td>1.000000</td>
      <td>0.939919</td>
      <td>0.856425</td>
      <td>0.895875</td>
    </tr>
    <tr>
      <th>2020-01-29</th>
      <td>0.569397</td>
      <td>0.553068</td>
      <td>0.561720</td>
      <td>0.579453</td>
      <td>0.581286</td>
      <td>0.573749</td>
      <td>0.591798</td>
      <td>0.600469</td>
      <td>0.584356</td>
      <td>0.605523</td>
      <td>...</td>
      <td>0.817592</td>
      <td>0.799846</td>
      <td>0.802474</td>
      <td>0.800217</td>
      <td>0.911849</td>
      <td>0.936386</td>
      <td>0.939919</td>
      <td>1.000000</td>
      <td>0.877684</td>
      <td>0.915625</td>
    </tr>
    <tr>
      <th>2020-05-01</th>
      <td>0.544281</td>
      <td>0.525156</td>
      <td>0.532077</td>
      <td>0.547014</td>
      <td>0.546911</td>
      <td>0.542506</td>
      <td>0.556496</td>
      <td>0.563103</td>
      <td>0.550994</td>
      <td>0.573485</td>
      <td>...</td>
      <td>0.754191</td>
      <td>0.715706</td>
      <td>0.732941</td>
      <td>0.732389</td>
      <td>0.823599</td>
      <td>0.855574</td>
      <td>0.856425</td>
      <td>0.877684</td>
      <td>1.000000</td>
      <td>0.918039</td>
    </tr>
    <tr>
      <th>2020-07-31</th>
      <td>0.558448</td>
      <td>0.541229</td>
      <td>0.549317</td>
      <td>0.569355</td>
      <td>0.569013</td>
      <td>0.557334</td>
      <td>0.573791</td>
      <td>0.580910</td>
      <td>0.566483</td>
      <td>0.592939</td>
      <td>...</td>
      <td>0.779510</td>
      <td>0.739858</td>
      <td>0.750893</td>
      <td>0.753166</td>
      <td>0.861716</td>
      <td>0.893976</td>
      <td>0.895875</td>
      <td>0.915625</td>
      <td>0.918039</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>80 rows × 80 columns</p>
</div>



![png](output_26_1.png)



![png](output_26_2.png)



```python

```

### Containment N-gram scores


```python

```


```python
# containment function
def calculate_containment(df1, df2, ngram_size):
    text1 = df1.text
    text2 = df2.text
    counts = CountVectorizer(analyzer='word', ngram_range=(ngram_size, ngram_size))
    ngrams = counts.fit_transform([text1, text2])
    
    ngram_array = ngrams.toarray()
    intersect = np.amin(ngram_array, axis=0)
    common_ngrams = sum(intersect)
    
    len_ngram_a = sum(ngram_array[0])
    
    containment_score = 1.0 * common_ngrams / len_ngram_a
    
    return containment_score
```


```python
# 2-ngram containment
ngram = 2
rows = []
for j in range(len(df)):
    columns = []
    for i in range(len(df)):
        columns.append( calculate_containment( df.iloc[i], df.iloc[j], ngram ) )
    rows.append(columns)
containment_df = pd.DataFrame(rows, index = pd.to_datetime(df.index), columns = df.index )

# containment df
display(containment_df)

# containment heatmap
plt.title('Heatmap of 2-nrgam containment scores')
sns.heatmap(containment_df, cmap = 'Blues')
plt.show()


# plot of containment of previous reports to most recent report
plt.plot(containment_df.iloc[:,-1])
plt.title('2-gram containment of most recent 10-Q with previous 10-Qs', fontsize = 24 )
plt.show()
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1994-01-26</th>
      <th>1994-08-12</th>
      <th>1995-02-09</th>
      <th>1995-05-15</th>
      <th>1995-08-11</th>
      <th>1996-02-12</th>
      <th>1996-05-13</th>
      <th>1996-08-12</th>
      <th>1997-02-10</th>
      <th>1997-05-12</th>
      <th>...</th>
      <th>2017-08-02</th>
      <th>2018-02-02</th>
      <th>2018-05-02</th>
      <th>2018-08-01</th>
      <th>2019-01-30</th>
      <th>2019-05-01</th>
      <th>2019-07-31</th>
      <th>2020-01-29</th>
      <th>2020-05-01</th>
      <th>2020-07-31</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1994-01-26</th>
      <td>1.000000</td>
      <td>0.460490</td>
      <td>0.384475</td>
      <td>0.327031</td>
      <td>0.308386</td>
      <td>0.246951</td>
      <td>0.187904</td>
      <td>0.181636</td>
      <td>0.161316</td>
      <td>0.144627</td>
      <td>...</td>
      <td>0.071914</td>
      <td>0.066807</td>
      <td>0.066269</td>
      <td>0.063633</td>
      <td>0.078234</td>
      <td>0.082889</td>
      <td>0.080341</td>
      <td>0.082163</td>
      <td>0.082155</td>
      <td>0.078791</td>
    </tr>
    <tr>
      <th>1994-08-12</th>
      <td>0.669307</td>
      <td>1.000000</td>
      <td>0.629680</td>
      <td>0.552293</td>
      <td>0.520768</td>
      <td>0.415312</td>
      <td>0.327070</td>
      <td>0.317506</td>
      <td>0.276351</td>
      <td>0.245652</td>
      <td>...</td>
      <td>0.083721</td>
      <td>0.075901</td>
      <td>0.076209</td>
      <td>0.074451</td>
      <td>0.090909</td>
      <td>0.090645</td>
      <td>0.089318</td>
      <td>0.088483</td>
      <td>0.091948</td>
      <td>0.089045</td>
    </tr>
    <tr>
      <th>1995-02-09</th>
      <td>0.555776</td>
      <td>0.626249</td>
      <td>1.000000</td>
      <td>0.798874</td>
      <td>0.728056</td>
      <td>0.531504</td>
      <td>0.416031</td>
      <td>0.402460</td>
      <td>0.347780</td>
      <td>0.310936</td>
      <td>...</td>
      <td>0.086941</td>
      <td>0.080448</td>
      <td>0.080186</td>
      <td>0.076360</td>
      <td>0.094406</td>
      <td>0.095977</td>
      <td>0.094255</td>
      <td>0.099017</td>
      <td>0.100109</td>
      <td>0.097679</td>
    </tr>
    <tr>
      <th>1995-05-15</th>
      <td>0.536634</td>
      <td>0.623524</td>
      <td>0.906849</td>
      <td>1.000000</td>
      <td>0.860502</td>
      <td>0.582656</td>
      <td>0.458309</td>
      <td>0.443364</td>
      <td>0.389781</td>
      <td>0.348582</td>
      <td>...</td>
      <td>0.096601</td>
      <td>0.087093</td>
      <td>0.089795</td>
      <td>0.085587</td>
      <td>0.099213</td>
      <td>0.106156</td>
      <td>0.104578</td>
      <td>0.104635</td>
      <td>0.106638</td>
      <td>0.101997</td>
    </tr>
    <tr>
      <th>1995-08-11</th>
      <td>0.519472</td>
      <td>0.603542</td>
      <td>0.848402</td>
      <td>0.883347</td>
      <td>1.000000</td>
      <td>0.632114</td>
      <td>0.487962</td>
      <td>0.471968</td>
      <td>0.410380</td>
      <td>0.365737</td>
      <td>...</td>
      <td>0.094454</td>
      <td>0.085345</td>
      <td>0.087806</td>
      <td>0.083042</td>
      <td>0.100524</td>
      <td>0.105671</td>
      <td>0.103680</td>
      <td>0.103933</td>
      <td>0.105005</td>
      <td>0.100917</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2019-05-01</th>
      <td>0.112871</td>
      <td>0.084923</td>
      <td>0.090411</td>
      <td>0.088093</td>
      <td>0.085423</td>
      <td>0.069783</td>
      <td>0.073400</td>
      <td>0.073227</td>
      <td>0.071429</td>
      <td>0.069812</td>
      <td>...</td>
      <td>0.488372</td>
      <td>0.487583</td>
      <td>0.493704</td>
      <td>0.482024</td>
      <td>0.749126</td>
      <td>1.000000</td>
      <td>0.845601</td>
      <td>0.875702</td>
      <td>0.722524</td>
      <td>0.675661</td>
    </tr>
    <tr>
      <th>2019-07-31</th>
      <td>0.118152</td>
      <td>0.090372</td>
      <td>0.095890</td>
      <td>0.093725</td>
      <td>0.090517</td>
      <td>0.073848</td>
      <td>0.076923</td>
      <td>0.074943</td>
      <td>0.072499</td>
      <td>0.070765</td>
      <td>...</td>
      <td>0.499463</td>
      <td>0.497726</td>
      <td>0.500000</td>
      <td>0.487432</td>
      <td>0.840909</td>
      <td>0.913233</td>
      <td>1.000000</td>
      <td>0.889747</td>
      <td>0.726333</td>
      <td>0.692391</td>
    </tr>
    <tr>
      <th>2020-01-29</th>
      <td>0.077228</td>
      <td>0.057221</td>
      <td>0.064384</td>
      <td>0.059936</td>
      <td>0.057994</td>
      <td>0.045393</td>
      <td>0.049912</td>
      <td>0.050343</td>
      <td>0.047084</td>
      <td>0.045509</td>
      <td>...</td>
      <td>0.350626</td>
      <td>0.357118</td>
      <td>0.344930</td>
      <td>0.327076</td>
      <td>0.534091</td>
      <td>0.604460</td>
      <td>0.568671</td>
      <td>1.000000</td>
      <td>0.692601</td>
      <td>0.658392</td>
    </tr>
    <tr>
      <th>2020-05-01</th>
      <td>0.099670</td>
      <td>0.076748</td>
      <td>0.084018</td>
      <td>0.078842</td>
      <td>0.075627</td>
      <td>0.061314</td>
      <td>0.066647</td>
      <td>0.065789</td>
      <td>0.063938</td>
      <td>0.062902</td>
      <td>...</td>
      <td>0.373524</td>
      <td>0.353970</td>
      <td>0.363817</td>
      <td>0.348711</td>
      <td>0.527098</td>
      <td>0.643723</td>
      <td>0.599192</td>
      <td>0.893961</td>
      <td>1.000000</td>
      <td>0.826767</td>
    </tr>
    <tr>
      <th>2020-07-31</th>
      <td>0.096370</td>
      <td>0.074932</td>
      <td>0.082648</td>
      <td>0.076026</td>
      <td>0.073276</td>
      <td>0.059621</td>
      <td>0.062243</td>
      <td>0.060927</td>
      <td>0.058320</td>
      <td>0.056707</td>
      <td>...</td>
      <td>0.364937</td>
      <td>0.354320</td>
      <td>0.358184</td>
      <td>0.347121</td>
      <td>0.514860</td>
      <td>0.606883</td>
      <td>0.575853</td>
      <td>0.856742</td>
      <td>0.833515</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>80 rows × 80 columns</p>
</div>



![png](output_31_1.png)



![png](output_31_2.png)


