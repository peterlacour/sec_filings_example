import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, ImageColorGenerator, STOPWORDS
from PIL import Image  # class for custom masks
import re
import os
from .text_processing_class import textPreProcess

class wordCloud():
    # Instantiate class
    def __init__(self):
        '''
        Description:    class initialisation, starts webdriver and creates dataframe
        Inputs:         :no inputs
        Outputs:        :no outputs
        '''

        self.currentDirectory   = os.getcwd()
        self.tpp                = textPreProcess()

        # Create circle mask
        x, y = np.ogrid[:300, :300]
        self.mask = (x - 150) ** 2 + (y - 150) ** 2 > 130 ** 2
        self.mask = 255 * self.mask.astype(int)

    # --------------------------------------------------------------------------

    def create_word_cloud( self, dictionary, title, masks = 1, columns = 1, rows = 1 ):
        '''
        Description:    creates world cloud grid with a given title and mask
        Inputs:         :dictionary -> data to create word cloud with
                        :title -> title of word cloud
                        :masks -> name of picture to create masks, default is a circle
                        :columns and rows -> grid to be created
        Outputs:        :saves and displays wordcloud
        '''

        if type(masks) == type(""):
            maskName = masks
            masks, imageColors = self._create_mask( masks )

        else:
            masks = self.mask
            maskName = 'Circles'


        fig = plt.figure(figsize=(14, 14),facecolor = 'white')
        plt.axis("off")

        ax = []
        count = 0
        for key, value in dictionary.items():
            lemmatisedWords = self.tpp.concatenate_texts( value )

            # Remove punctuation
            for words in lemmatisedWords:
                words = re.sub('[\W]+', ' ', words )

            img = WordCloud( max_font_size = 35, background_color="white",
                                    mask = masks, width = 512, height = 512,
                                    #contour_width = 0.25, contour_color='seagreen',
                                    stopwords = STOPWORDS ).generate( lemmatisedWords )

            # create subplot and append to ax
            if title.lower() == "date":
                title2 = 'Month Year ' + key[:2] + "/" + key[2:]
            elif title.lower() == "company":
                title2 =  key

            ax.append( fig.add_subplot(rows, columns, count+1 ) )
            ax[-1].set_title( title2 , fontdict = {'size': 18, 'fontweight': 'bold'} )  # set title
            #plt.imshow(img.recolor(color_func = imageColors), interpolation="bilinear" )
            plt.imshow( img, interpolation = "bilinear" )

            plt.axis("off")
            plt.tight_layout( pad = 0 )

            # do extra plots on selected axes/subplots
            # note: index starts with 0
            count += 1
        plt.tight_layout(pad=0)
        plt.show()
        #fig.savefig( self.currentDirectory + '/Figures/wordcloud_' + maskName + '.png', dpi = 300)

    # --------------------------------------------------------------------------

    def _transform( self, pixel ):
        '''
        Description:    Used to transform picture array to be able to used as word cloud
        Inputs:         :pixel
        Outputs:        :transformed pixel
        '''

        if pixel == 0.00:
            return 255
        else:
            return pixel

    # --------------------------------------------------------------------------

    def _create_mask( self, pictureName ):
        '''
        Description:    Creates custom masks given a picture name
        Inputs:         :pictureName
        Outputs:        :returns mask
        '''
        mask = np.array(Image.open( pictureName + '.png'))
        imageColors = ImageColorGenerator(mask)
        # get_single_color_func('deepskyblue')
        mask = np.ndarray((mask.shape[0],mask.shape[1]), np.int32)

        for i in range(len(mask)):
            mask[i] = list(map(self._transform, mask[i]))

        return mask, imageColors
