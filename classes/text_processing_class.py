
# Load packages
import numpy as np
import os
import re

import nltk
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords as nltk_stopwords
from ast import literal_eval


class textPreProcess():

    def __init__( self ):
        '''
        Class description....
        '''

        # Get current directory
        self.currentDirectory = os.getcwd()

        # Load nltk libraries
        nltk.download('wordnet', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)

        # Load stop words
        #self.stopwords = list( pd.read_csv( self.currentDirectory + '/Custom_Stopwords.csv' )["Stop Words"] )
        self.stopwords = nltk_stopwords.words('english')
        # Default ticker and text column
        self.textColumn = 'text'

    # --------------------------------------------------------------------------

    def lemmatize( self, documentsDf ):
        '''
        Description:    Lemmatizes text
        Inputs:         :documents dataframe and column name of documents
        Outputs:        :documents dataframe with column containing the lemmatised text
        '''

        documentsDf['lemmatised_text'] = np.nan
        texts = [entry.lower() for entry in documentsDf[ self.textColumn ] ]
        texts = [ word_tokenize(entry) for entry in texts ]
        tag_map = defaultdict( lambda : wn.NOUN )
        tag_map['J'] = wn.ADJ
        tag_map['V'] = wn.VERB
        tag_map['R'] = wn.ADV

        for idx, entry in enumerate( texts ):
            final_words = []
            lemmatizer = WordNetLemmatizer()
            for word, tag in pos_tag( entry ):
                if word not in self.stopwords and word.isalpha():
                    lemmatised_word = lemmatizer.lemmatize(word, tag_map[tag[0]])
                    final_words.append(lemmatised_word)
            final_words = [ w for w in final_words if w not in self.stopwords ]

            documentsDf.loc[documentsDf.index[idx], 'lemmatised_text'] = str(final_words)

        return documentsDf

    # --------------------------------------------------------------------------

    def delete_stopwords( self, document ):
        '''
        Description:    deletes stopwords
        Inputs:         :document
        Outputs:        :document without stopwords
        '''
        text = [ word.strip() for word in document.split()]
        words = [ w for w in text if w not in self.stopwords ]
        return ' '.join(words)

    # --------------------------------------------------------------------------

    def concatenate_texts( self, arrayOfTexts, notArrayOfTexts = True, concatenateType = "all" ):
        '''
        Description:    concatenates text
        Inputs:         :
        Outputs:        :
        '''
        if notArrayOfTexts == True:
            arrayOfTexts = [ literal_eval( t ) for t in arrayOfTexts ]

        if concatenateType.lower() == "all":
            text = ''
            for texts in arrayOfTexts:
                for t in texts:
                    text += t + ' '
        elif concatenateType.lower() == "single":
            text = ''
            for t in arrayOfTexts:
                    text += t + ' '
        return text

    # --------------------------------------------------------------------------

    def back_to_list( self, documentsDf ):
        #documentList = []
        for i, d in enumerate( documentsDf[ self.textColumn ].values ):
            documentsDf[ self.textColumn ].loc[ i ] = literal_eval( d )
        #documentList.append( [ literal_eval( t ) for t in documentsDf[ column ] ] )
        return documentsDf

    # --------------------------------------------------------------------------

    def stem( self, documentsDf ):
        '''
        Description:    ....

        '''
        porter = PorterStemmer()

        documentsDf['stemmed_text'] = np.nan

        texts = [ entry.lower() for entry in documentsDf[ self.textColumn ] ]
        texts = [ w for w in texts if w not in self. stopwords ]

        #
        for i, entry in enumerate( texts ):
            # entry = word_tokenize( entry )
            entry                               = ( re.sub('[\W]+', ' ', entry ) )      #remove non-word characters and make text lowercase
            entry                               = ( re.sub('[\d]+', '', entry) ) #to remove numbers [0-9]
            entry                               = ( re.sub('\n', ' ', entry) )
            stopwordsDeleted                    = self._delete_stopwords( entry )
            documentsDf['stemmed_text'].loc[ i ] = str( [ porter.stem( word.strip() ) for word in stopwordsDeleted.split() ] )

        return documentsDf
