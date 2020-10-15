import numpy as np
import pandas as pd
import re
import requests
from bs4 import BeautifulSoup
from ratelimit import limits, sleep_and_retry


class SecCall(object):

    def __init__(self):
        pass

    # --------------------------------------------------------------------------

    SEC_CALL_LIMIT = {'calls': 10, 'seconds': 1}
    @staticmethod
    @sleep_and_retry
    @limits(calls=SEC_CALL_LIMIT['calls'] / 2, period=SEC_CALL_LIMIT['seconds'])
    def _call_sec(url):
        return requests.get(url)

    # --------------------------------------------------------------------------

    def get(self, url):
        return self._call_sec(url).text


class SecData(SecCall):

    def __init__(self):
        self.start_date = '1980-01-01' #'1999-01-01'
        self.sec_api = SecCall()

    # --------------------------------------------------------------------------

    def get_sec_filings(self, cik, doc_type, start=0, count=1000):
        oldest_data = pd.to_datetime(self.start_date)
        rss_url = 'https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany' \
            '&CIK={}&type={}&start={}&count={}&owner=exclude&output=atom' \
            .format(cik, doc_type, start, count)
        sec_data = self.sec_api.get(rss_url)
        feed = BeautifulSoup(sec_data.encode('ascii'), 'xml').feed
        entries = [
            (
                entry.content.find('filing-href').getText(),
                entry.content.find('filing-type').getText(),
                entry.content.find('filing-date').getText())
            for entry in feed.find_all('entry', recursive=False)
            if pd.to_datetime(entry.content.find('filing-date').getText()) >= oldest_data ]

        return entries

    # --------------------------------------------------------------------------

    def download_filing(self, link):
        return self.sec_api.get(link)

    # --------------------------------------------------------------------------

    def get_documents(self, text):

        # get list of starts and ends of document tag
        doc_start_is = [x.end() for x in re.compile(r'<DOCUMENT>').finditer(text)]
        doc_end_is = [x.start() for x in re.compile(r'</DOCUMENT>').finditer(text)]

        # iterate through the documents and append to extracted documents to list
        extracted_docs = [ text[doc_start_i:doc_end_i] for doc_start_i, doc_end_i in zip(doc_start_is, doc_end_is)]

        return extracted_docs

    # --------------------------------------------------------------------------

    def get_document_type(self, doc):
        return [x[len('<TYPE>'):] for x in re.compile(r'<TYPE>[^\n]+').findall(doc)][0].lower()

    # --------------------------------------------------------------------------

    def remove_html_tags(self, text):
        text = BeautifulSoup(text, 'html.parser').get_text()

        return text

    # --------------------------------------------------------------------------

    def clean_text(self, text):
        #text = text.lower()
        text = self.remove_html_tags(text)

        return text

    # --------------------------------------------------------------------------

    def get_10qs(self, cik):

        data = self.get_sec_filings(cik, '10-Q')
        doc_type = '10-Q'
        filings = {}
        for index_url, file_type, file_date in data:
            if (file_type == doc_type):
                file_url = index_url.replace('-index.htm', '.txt').replace('.txtl', '.txt')
                filings[file_date] = self.get(file_url)

        filings_documents = {}
        for file_date, filing in filings.items():
            filings_documents[file_date] = self.get_documents(filing)

        ten_qs = {}
        for file_date, documents in filings_documents.items():
            for document in documents:
                if doc_type.lower() in self.get_document_type(document) and 'A' not in self.get_document_type(document):
                    ten_qs[file_date] = document
        return ten_qs

    # --------------------------------------------------------------------------

    def get_mdna(self, doc):
        # get management discussion and analysis section

        # start
        # Item 2. Management’s Discussion and Analysis of Financial Condition and Results of Operations
        # and not item 3
        doc = BeautifulSoup(doc, 'html.parser').get_text()

        doc = doc.replace('&nbsp;', ' ').replace('&#160;', ' ').replace(u'\xa0', u' ')

        # trial rn, slows down script but most recent tsla report splits words with html tags...

        #item2_start = [x.end() for x in re.compile(r'item 2').finditer(doc.lower())]
        item2_start = [x.end() for x in re.compile(r'item[\n| |  ]*2').finditer(doc.lower())]

        #if len(item2_start) == 0:
        #    item2_start = [x.end() for x in re.compile(r'item  2').finditer(doc.lower())]

        #criteria1 = [x.end() for x in re.compile(r'item[\n| |  ]*3').finditer(doc.lower())]
        #if len(criteria1) != 0:
        #    item2_start = [ item  for item in item2_start for crit in criteria1 if abs(item - crit) > 10000 ]
        bool = True
        line = 0
        while bool:
            try: # catchin index error and assigning last value
                start = item2_start[line]

                #start = item2_start[line]
                criteria1 = [x.end() for x in re.compile(r'item[\n| |  ]*\d').finditer(doc[start+len('item'):start+200].lower())] # 500 or 1000 or 2000 before
                #criteria2 = [x.end() for x in re.compile(r'item[\n| |  ]*\d').finditer(doc[start+len('item'):start+2000].lower())]
                #print(len(criteria1), len(criteria2))
                criteria2 = [x.end() for x in re.compile(r'extraordinary').finditer(doc[start-100:start+200].lower())] #2000

                #criteria2 = [x.end() for x in re.compile(r'item[\n| |  ]*\d').finditer(doc[start+len('item'):start+2000].lower())]
                #print(len(criteria1), len(criteria2))
                criteria3 = [x.end() for x in re.compile(r'part i[^î]').finditer(doc[max(start-10000,0):start].lower())] #1000 #2000 # start-20000
                criteria4 = [x.end() for x in re.compile(r'management').finditer(doc[start-200:start+200].lower())] #1000 #2000

                if len(criteria1) == 0 and len(criteria2) == 0 and len(criteria3) == 0  and len(criteria4) != 0:
                    bool = False
                line += 1

            except:
                line = 0
                bool2 = True
                while bool2:
                    try: # catchin index error and assigning last value
                        start = item2_start[line]

                        criteria1 = [x.end() for x in re.compile(r'item[\n| |  ]*\d').finditer(doc[start+len('item'):start+200].lower())] #1000 #2000
                        criteria2 = [x.end() for x in re.compile(r'extraordinary').finditer(doc[start-100:start+200].lower())] #2000
                        if len(criteria1) == 0 and len(criteria2) == 0:
                            bool2 = False
                        line += 1
                    except:
                        start = item2_start[-1]
                        bool2 = False

                bool = False

        #line = 0
        #start = item2_start[0]
        #while 'item 3' in doc.lower()[start:start+2000] or r'item ' in doc.lower()[start+len('item'):start+2000]: #or 'part ii' in doc.lower()[start:start+2000]: # 'management' in doc.lower()[start:start+10000] and #or 'part ii' in doc.lower()[start:start+2000]):
        #    line += 1
        #    start = item2_start[line]

        # end
        # either item 3 and quantitative
        # or part II and item I legal
        #item2_end1 = [x.start() for x in re.compile(r'item\n3').finditer(doc[start:].lower()) if x.end() > 10000 ]
        item2_end1 = [x.end() for x in re.compile(r'item[\n| |  ]*3').finditer(doc[start:].lower()) if x.end() > 10000 ]
        # item2_end2 = [x.start() for x in re.compile(r'part.{9,}ii').finditer(doc[start:].lower()) if x.end() > 10000 ]
        item2_end2 = [x.end() for x in re.compile(r'part[\n| |  ]*ii').finditer(doc[start:].lower()) if x.end() > 10000]

        line = 0
        if len(item2_end1) != 0:
            end1 = item2_end1[0]
            while 'quantitative' in doc.lower()[end1:end1+500] and 'item 3' in doc.lower()[end1:end1+500] and end1 < 10000:
                line += 1
                end1 = item2_end1[line]
        else:
            end1 = 100000000

        if len(item2_end2) != 0:
            line = 0
            end2 = item2_end2[line]
            try:
                while 'item 1' not in doc.lower()[end2:end2+2000]: #and 'other information' in doc.lower()[end2:end2+500]:
                    line += 1
                    end2 = item2_end2[line]
            except:
                    end2 = item2_end2[-1]
        else:
            end2 = 100000000
        end = min(end1, end2) + start

        if '</font>' in doc.lower():
            temp = doc[start:end].split('</font>')
            management_discussion = ' \n '.join( r for r in [ self.clean_text( t ) for t in temp if '<td' not in t ] if len(str(r)) > 50 )
        else:
            temp = doc[start:end].split('</div>')
            management_discussion = ' \n '.join( r for r in [ self.clean_text( t ) for t in temp if '<td' not in t ] if len(str(r)) > 50 )
            if len(management_discussion) < 150:
                # choose between span or div?
                temp = doc[start:end].split('</span>')
                management_discussion = ' \n '.join( r for r in [ self.clean_text( t ) for t in temp if '<td' not in t ] if len(str(r)) > 50 )
            if len(management_discussion) < 150:
                temp = doc[start:end].split('</p>')
                management_discussion = ' \n '.join( r for r in [ sec.clean_text( t ) for t in temp if '<td' not in t ] if len(str(r)) > 50 )

        return management_discussion


'''
from tqdm.notebook import tqdm

sec = SecData()
cik_dict = { 'AAPL': '0000320193', 'XOM': '0000034088', 'TSLA': '0001318605', 'JNJ': '0000200406' }
#cik_dict = { 'AAPL': '0000320193' }
ten_qs = {}
for ticker, cik in tqdm(cik_dict.items()):
    ten_qs[ticker] = sec.get_10qs( cik )

ticker = 'TSLA'
mdna = {}
for file_date, doc in tqdm(ten_qs[ticker].items()):
    print(file_date)
    mdna[file_date] = sec.get_mdna(doc)
df = pd.DataFrame(mdna, index = ['MDnA']).transpose()



#file_date = '2015-05-11'
#df.index[0]
for i, d in enumerate(df.MDnA):
    print(df.index[i], d[:50])




df.head(100)

2006-12-29
2006-05-05


doc = doc1

doc = ten_qs[ticker]['2006-12-29']
#doc = sec.get_mdna(doc)
doc = BeautifulSoup(doc, 'html.parser').get_text()
#mdna[file_date] = sec.get_mdna(doc)


doc = doc.replace('&nbsp;', ' ').replace('&#160;', ' ').replace(u'\xa0', u' ')

#item2_start = [x.end() for x in re.compile(r'item 2').finditer(doc.lower())]
item2_start = [x.end() for x in re.compile(r'item[\n| |  ]*2').finditer(doc.lower())]
start = item2_start[0]

#if len(item2_start) == 0:
#    item2_start = [x.end() for x in re.compile(r'item  2').finditer(doc.lower())]



#criteria1 = [x.end() for x in re.compile(r'item[\n| |  ]*3').finditer(doc.lower())]
#if len(criteria1) != 0:
#    item2_start = [ item  for item in item2_start for crit in criteria1 if abs(item - crit) > 10000 ]
bool = True
line = 0
while bool:
    try: # catchin index error and assigning last value
        start = item2_start[line]

        criteria1 = [x.end() for x in re.compile(r'item[\n| |  ]*\d').finditer(doc[start+len('item'):start+200].lower())] #1000 #2000
        criteria2 = [x.end() for x in re.compile(r'extraordinary').finditer(doc[start-100:start+200].lower())] #2000
        #criteria3 = [x.end() for x in re.compile(r'part i[^i]').finditer(doc[start+len('item'):start+500].lower())] #1000 #2000
        #criteria4 = [x.end() for x in re.compile(r'part ii').finditer(doc[start+len('item'):start+500].lower())] #1000 #2000
        #criteria2 = [x.end() for x in re.compile(r'item[\n| |  ]*\d').finditer(doc[start+len('item'):start+2000].lower())]
        criteria3 = [x.end() for x in re.compile(r'part i[^î]').finditer(doc[max(start-10000,0):start].lower())] #1000 #2000
        criteria4 = [x.end() for x in re.compile(r'management').finditer(doc[start-200:start+200].lower())] #1000 #2000
        print(start)
        print(len(criteria1), len(criteria2), len(criteria3), len(criteria4))
        if len(criteria1) == 0 and len(criteria2) == 0 and len(criteria3) == 0 and len(criteria4) != 0:
            bool = False
        line += 1

    except:
        line = 0
        bool2 = True
        while bool2:
            try: # catchin index error and assigning last value
                start = item2_start[line]
            except:
                start = item2_start[-1]
                bool2 = False
            criteria1 = [x.end() for x in re.compile(r'item[\n| |  ]*\d').finditer(doc[start+len('item'):start+200].lower())] #1000 #2000
            criteria2 = [x.end() for x in re.compile(r'extraordinary').finditer(doc[start-100:start+200].lower())] #2000

            if len(criteria1) == 0 and len(criteria2) == 0:
                bool2 = False
            line += 1
        bool = False
        print('done')

start

doc[109309+len('item'):109309+1000]
#doc[174641:]

#line = 0
#start = item2_start[0]
#while 'item 3' in doc.lower()[start:start+2000] or r'item ' in doc.lower()[start+len('item'):start+2000]: #or 'part ii' in doc.lower()[start:start+2000]: # 'management' in doc.lower()[start:start+10000] and #or 'part ii' in doc.lower()[start:start+2000]):
#    line += 1
#    start = item2_start[line]
#start = item2_start[2]
#start
#[x.end() for x in re.compile(r'part i[^î]').finditer(doc[max(start-1000,0):start].lower())]

#doc[max(start-10000,0):start].lower()
# end
# either item 3 and quantitative
# or part II and item I legal
#item2_end1 = [x.start() for x in re.compile(r'item\n3').finditer(doc[start:].lower()) if x.end() > 10000 ]
item2_end1 = [x.end() for x in re.compile(r'item[\n| |  ]*3').finditer(doc[start:].lower()) if x.end() > 10000 ]
# item2_end2 = [x.start() for x in re.compile(r'part.{9,}ii').finditer(doc[start:].lower()) if x.end() > 10000 ]
item2_end2 = [x.end() for x in re.compile(r'part[\n| |  ]*ii').finditer(doc[start:].lower()) if x.end() > 10000]

line = 0
if len(item2_end1) != 0:
    end1 = item2_end1[0]
    while 'quantitative' in doc.lower()[end1:end1+500] and 'item 3' in doc.lower()[end1:end1+500] and end1 < 10000:
        line += 1
        end1 = item2_end1[line]
else:
    end1 = 100000000

if len(item2_end2) != 0:
    line = 0
    end2 = item2_end2[line]
    try:
        while 'item 1' not in doc.lower()[end2:end2+2000]: #and 'other information' in doc.lower()[end2:end2+500]:
            line += 1
            end2 = item2_end2[line]
    except:
            end2 = item2_end2[-1]
else:
    end2 = 100000000
end = min(end1, end2) + start

if '</font>' in doc.lower():
    temp = doc[start:end].split('</font>')
    management_discussion = ' \n '.join( r for r in [ sec.clean_text( t ) for t in temp if '<td' not in t ] if len(str(r)) > 50 )
else:
    temp = doc[start:end].split('</div>')
    management_discussion = ' \n '.join( r for r in [ sec.clean_text( t ) for t in temp if '<td' not in t ] if len(str(r)) > 50 )
    if len(management_discussion) < 150:
        # choose between span or div?
        temp = doc[start:end].split('</span>')
        management_discussion = ' \n '.join( r for r in [ sec.clean_text( t ) for t in temp if '<td' not in t ] if len(str(r)) > 50 )
    # second check
    if len(management_discussion) < 150:
        temp = doc[start:end].split('</p>')
        management_discussion = ' \n '.join( r for r in [ sec.clean_text( t ) for t in temp if '<td' not in t ] if len(str(r)) > 50 )
management_discussion




#[23606, 31734, 35671]

doc[23606:end]

#mdna = {}
#for file_date, doc in tqdm(ten_qs[ticker].items()):
#    print(file_date)
#    mdna[file_date] = sec.get_mdna(doc)
#df = pd.DataFrame(mdna, index = ['MDnA']).transpose()
#display(df.head())
'''
