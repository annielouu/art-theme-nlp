import streamlit as st
from plotly.express.data import stocks
import plotly.express as px
import plotly.graph_objects as go
import base64

st.set_page_config(page_title="Evolution of Themes in Art", layout="wide")

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@700&display=swap');
        
        /* Apply Poppins bold and font size 18 to markdown with the custom class */
        .custom-markdown {
            font-family: 'Poppins', sans-serif;
            font-weight: 700; /* Bold */
            font-size: 18px;
            margin-left: 20px;
            margin-right: 20px;
            margin-top: 20px;
            margin-bottom: 10px;
        }
        
        /* Center text alignment */
        .center-text {
            text-align: center;
            margin-bottom: 10px;
            margin-left: 20px;
            margin-right: 20px;
        }

        /* Background color for specific columns */
        .col1-background {
            background-color: #FFD700;  /* Gold color for column 1 */
        }
        
        .col3-background {
            background-color: #ADD8E6;  /* Light blue color for column 3 */
        }

        .boxed-content {
            border: 2px solid #E0E0E0;  /* Light gray border */
            padding-top: 20px;             /* Padding for the content */
            padding-left: 70px;
            padding-right: 70px;
            padding-bottom: 20px;
            margin: 20px;              /* Margin around the box */
            border-radius: 10px;       /* Rounded corners */
        }

        .boxed-content-2 {
            border: 2px solid #E0E0E0;  /* Light gray border */
            padding-top: 20px;             /* Padding for the content */
            padding-left: 40px;
            padding-right: 40px;
            padding-bottom: 20px;
            margin: 10px;              /* Margin around the box */
            border-radius: 10px;       /* Rounded corners */
        }

    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
        .key-words-text {
            color: #CB402E;  /* red color text */;
            font: 
        }
    </style>
    """,
    unsafe_allow_html=True,
)

custom_css = """
<style>
    .text-block {
        padding: 20px;
        background-color: #f0ebeb;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

st.sidebar.markdown("""
# Content
- [Overview](#overview)
- [About the Data](#about-the-data)
- [Preprocessing](#preprocessing)
- [Word Frequency Analysis](#word-frequency-analysis)
- [Topic Modeling Using BERT](#topic-modeling-using-bert)
- [Dimensionality Reduction + Clustering](#dimensionality-reduction-clustering)
- [Conclusion](#conclusion)
""", unsafe_allow_html=True)


df = stocks(indexed=False, datetimes=True)

st.title('Evolution of Themes in Art')

st.header('Overview')
# with open('myfile.html', 'r') as f:
#     content = f.read()
#     st.components.v1.html(content, height=400)

st.markdown("""
Art reflects society, politics, culture, and personal experiences. The advancements in machine learning techniques offered us more powerful and structured ways to explore art history.

In this project, I used machine learning and Natural Language Processing to analyze the evolution of themes in art history, from 3050 BCE to 2021 CE. 
            
The project explores three approaches:
  1. Word Frequency Analysis
  2. Topic Modeling
  3. Dimensionality Reduction and Clustering
""")

st.header('About the Data')
st.markdown("""
            The dataset contains 124,170 artworks from 3052 scraped from Wikiart. These artworks represent creations from 3,052 artists and span from\
            3050 BCE to 2021 CE.
            Columns include artwork style, title, artist, date of creation, and link to the piece.""")
st.markdown("""
            This data was collected and contributed to Kaggle by Antoine Gruson. More details can be found [here](https://www.kaggle.com/datasets/antoinegruson/-wikiart-all-images-120k-link).""")


st.header('Preprocessing')

with st.expander("Tokenization & Stop Words Removal"):
    st.markdown("""
    <ul>
        <li>Use Natural Language Toolkit (NLTK) library to handle stopwords across multiple languages</li>
        <li>Tokentize text using TextBlob</li>
    </ul>""",
    unsafe_allow_html=True)
    code = '''
    languages = ['english', 'german', 'french', 'italian', 'spanish', 'greek', 'romanian', 'chinese', 'russian']
    stops = set()

    for lang in languages:
        stops.update(stopwords.words(lang))

    def remove_stop_words(text):
        text = text.lower()
        original_text = text
        text = TextBlob(text).words #Tokentize
        text = [w for w in text if not w in stops]
        if not text:  # If the list is empty after removing stop words, let's not remove the stop word since it might be meaningful information
            return original_text
        text = " ".join(text)
        return text
    
    wikiart['Artwork'] = wikiart['Artwork'].apply(remove_stop_words)
    '''
    st.code(code, language='python')

with st.expander("Lemmatization"):
    st.markdown("""
    <ul>
        <li>Transform words to their original form for analysis (i.e. running -> run, better -> good) </li>
    </ul>""",
    unsafe_allow_html=True)
    
    code = '''
    lemmatizer = WordNetLemmatizer()

    def lemmatize_text(text):
        words = TextBlob(text).words
        lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
        return " ".join(lemmatized_words)
    
    wikiart['Artwork'] = wikiart['Artwork'].apply(lemmatize_text)
    '''
    st.code(code, language='python')

st.header('Word Frequency Analysis')

with st.expander("Read Methodology"):
    st.markdown("""
    <ol>
        <li>Calculated the frequency of each word and word combination present in artwork titles</li>
        <li>Ranked words and combinations based on frequency</li>
        <li>Identified the top 30 words and phrases across history</li>
        <li>Analyzed frequency trends of selected words</li>
    </ol>""",
    unsafe_allow_html=True)
    st.text('Words with highest frequecies')
    code = """
    word_freq = Counter(wikiart['Artwork']) # Caldulate word frequency
    most_common_words = word_freq.most_common(30)  # Get the top 30 words"""
    st.code(code, language='python')

    st.text("""Word frequency over time""")
    code = """
def landscape_count(df):
    return df['Artwork'].str.lower().str.contains('landscape').sum()

word_counts_by_date = wikiart.groupby('Date').apply(landscape_count).reset_index()
word_counts_by_date.columns = ['Date', 'Word_Count']
fig = px.bar(
    word_counts_by_date,'
    x='Date',
    y='Word_Count',
    title='Frequency of "Landscape" Over Time',
    template='plotly_white'
)
fig.update_traces(marker_line_width=0.5, marker_line_color="#EE6352")
fig.update_layout(width=1200, height=400, margin=dict(l=20, r=50, t=50, b=20), yaxis_title = 'Count')
"""
    st.code(code, language = 'python')
    

# with st.container():
#   st.markdown("""
#               <p class="text-block">
#   Methodology<br>
#   1. Calculated the frequency of each word and word combination present in artwork titles<br>
#   2. Ranked words and combinations based on frequency<br>
#   3. Identified the top 15 words and phrases across history<br>
#   4. Analyzed frequency trends of selected words<br>
#              </p>""",
#              unsafe_allow_html=True)

st.subheader("""What are the most popular words and phrases?""")    

st.markdown("""
  Across history, the most frequent word used in artwork titles is **"Untitled"**. This does not mean the titles are missing in the dataset. In art, it is a common practice to intentionally title\
            artworks as "Untitled" to encourage open interpretation.""")
st.markdown("""
            **"Self portrait"** has the second highest frequency, followed by **"Composition"**.""")
        
# Top 15 Words and Phrases
words = ['Untitled','Composition','Self Portrait', 'Landscape', 'Still Life','Portrait of a Man','Madonna and Child','Crucifixion','Portrait of a Woman','Annunciation','Spring','Nude','Flowers','Adoration of the Magi','Hamlet']
values = [4797,415,281+266+244, 203, 173, 117,102,88,85,79+78,66,66,63,58,58]
sorted_indices = sorted(range(len(values)), key=lambda k: values[k])
sorted_words = [words[i] for i in sorted_indices]
sorted_values = [values[i] for i in sorted_indices]
fig = go.Figure(data=[go.Bar(y=sorted_words, x=sorted_values, orientation='h', marker_color='#EE6352')])
fig.update_layout(
    height = 550,
    bargap=0.2,
    title={
    'text': 'Top 15 Words & Phrases in Artwork Titles Across History',
    'font': {
        'size':18
        }
    },
    xaxis_title='Frequency',
    xaxis={
        'title_font': {
            'size': 18  # Adjust x-axis title size here
        },
        'tickfont': {
            'size': 14  # Adjust x-axis tick text size here
        }
    },
    yaxis_title='Words',
    yaxis={
        'title_font': {
            'size': 18  # Adjust y-axis title size here
        },
        'tickfont': {
            'size': 14  # Adjust y-axis tick text size here
        }
    }
)
st.plotly_chart(fig)

st.subheader("""How does word frequency change over time?""")

tab1, tab2, tab3, tab4, tab5, tab6, tab7= st.tabs(["Untitled","Self Portrait","Landscape","Man","Woman", "Annunciation","Crucifixion"])
with tab1:
  with open('plots/untitled_frequency.html', 'r') as f:
      content = f.read()
      st.components.v1.html(content,height=400)
with tab2:
  with open('plots/self_portrait_frequency.html', 'r') as f:
      content = f.read()
      st.components.v1.html(content,height=400)
with tab3:
  with open('plots/landscape_frequency.html', 'r') as f:
      content = f.read()
      st.components.v1.html(content,height=400)
with tab4:
  with open('plots/men_frequency.html', 'r') as f:
      content = f.read()
      st.components.v1.html(content,height=400)
with tab5:
  with open('plots/woman_frequency.html', 'r') as f:
      content = f.read()
      st.components.v1.html(content,height=400)
with tab6:
  with open('plots/annunciation_frequency.html', 'r') as f:
      content = f.read()
      st.components.v1.html(content,height=400)
with tab7:
  with open('plots/crucifixion_frequency.html', 'r') as f:
      content = f.read()
      st.components.v1.html(content,height=400)\
      
st.markdown("""
- Although <span style='color:#D85645; font-weight:bold'>"untitled"</span> is the most frequently used word in artwork titles, \
            it only became popular in the **1900s**, likely due to contemporary art's shift toward abstraction.
- <span style='color:#D85645; font-weight:bold'>"Self-portrait"</span> titles peaked in the **1630s-1640s** and again from the **late 19th to early 20th century**.
- <span style='color:#D85645; font-weight:bold'>"Landscape"</span> experienced a minor peak during the **1620s-1650s** and a major surge between **1900-1920s**.
- The term <span style='color:#D85645; font-weight:bold'>"man"</span> appeared as early as **2035 BCE**, with significant peaks in **1756** and **1910**.
- <span style='color:#D85645; font-weight:bold'>"Woman"</span> first appeared in **2035 BCE**, showing sporadic early use with notable peaks in **1550** and **1912**.
- Unlike other terms, <span style='color:#D85645; font-weight:bold'>"Annunciation"</span> and <span style='color:#D85645; font-weight:bold'>"Crucifixion"</span> exhibited a relatively even distribution between **1000s and 2000s**. Both of them peaked between 15th and 16th century.
""", unsafe_allow_html=True)

st.divider()

st.header('Topic Modeling Using BERT')

with st.expander("Read Methodology"):
  st.markdown("""
              <ul>
              <li> BERTopic is a tool that turns each title into embeddings (vectors of real numbers) that represent semantic information. \
              Titles with similar meanings tend to have similar vectors. BERTopic then groups these embeddings to find patterns, \
              and label each group with the words that best describe the topic of that group.</li>
              <li> I used BERTopic to perform topic modeling on all 124170 artwork titles.</li>
              <li>I repeated the process on artwork titles within specific time periods.</li>
              <li> Note: BERT is an expensive model and the data set is very big. Running the BERTopic model on the entire data set crashed my computer several times, so I used sampling in early analysis to test ideas.\
              Once I chose an approach, I used a third party server to run the final analysis on the entire dataset.</li>
              </ul>""",
             unsafe_allow_html=True)
  code = '''
    topic_model = BERTopic(language = 'multilingual', top_n_words = 10, verbose=True, nr_topics=12, min_topic_size=6)
    topics, probabilities = topic_model.fit_transform(corpus)
    topic_model.visualize_barchart(top_n_topics = 10).show()
    '''
  st.code(code,language='python')
  st.text("""To get topic modeling results from any specific period of time, I have defined the following function.""")
  code = '''
    def get_topics(
        min_year=min(wikiart['Date']),
        max_year=max(wikiart['Date']),
        top_n_words=10,
        n_gram_range=(1,1),
        min_topic_size =10,
        top_n_topics = 5
    ):
        """
        First, we create a function that takes the following arguments:

        :params:
        min_year: minimum year. default = - 3050
        max_year: maximum year, default = 2022
        top_n_words: the number of words per topic to be extracted. 
        n_gram_range: the CountVectorizer used when creating the topic representation
        top_n_topics: how many topics to be displayed in final result

        :returns:
        Outputs are number of artworks in the timeframe,
        topic modeling results,
        topic_model and corpus selected (to pass on to the evaluation function)
        """
        wikiart_selected = wikiart[(wikiart['Date'] >= min_year) & (wikiart['Date'] < max_year)]
        corpus_selected = wikiart_selected['Artwork']
        topic_model = BERTopic(language = 'multilingual', top_n_words = top_n_words, n_gram_range=n_gram_range,min_topic_size =min_topic_size, verbose=True)
        topics, probabilities = topic_model.fit_transform(corpus_selected)
        print(f'There are {len(wikiart_selected)} artworks in this timeframe.')
        topic_model.visualize_barchart(top_n_topics = top_n_topics).show()

        return topic_model, wikiart_selected['Artwork']
    '''
  st.code(code,language='python')
with st.expander("Read Result: Topic Word Scores"):
  st.markdown("""
              <ul>
              <li> The graph below displays the raw output from the topic modeling process. Each topic is represented by a set of words,\
               where the importance of each word is determined based on its frequency and its similarity to the topic's centroid in the embedding space.</li>
              <li> Human-readable topic labels are created and interpreted based on raw output. </li>
              </ul>""",
             unsafe_allow_html=True)
  st.image('./plots/all.png')

st.subheader('8 Most Common Themes in Art Across Time')

st.markdown("""Through topic modeling on all 124,170 artwork titles, I have identified 8 common themes in artworks across time. These are **self portrait,\
             water bodies & transportation, nude bathers, animals, seated or resting figures, games and entertainment, fire, and body parts**. 
            """)

def image_to_base64(img_path):
    with open(img_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')
    

col1, col2 = st.columns(2)

encoded_image1 = image_to_base64("images/self_portrait.jpg")
encoded_image2 = image_to_base64("images/the-sea-port-in-amsterdam.jpg")
encoded_image3 = image_to_base64("images/bath.jpg")
encoded_image4 = image_to_base64("images/horse.jpg")
encoded_image5 = image_to_base64("images/chair.jpg")
encoded_image6 = image_to_base64("images/player.jpg")
encoded_image7 = image_to_base64("images/fire.jpg")
encoded_image8 = image_to_base64("images/hands.jpg")

with col1:
   st.markdown(
      f"""
      <div class="boxed-content">
          <div class="custom-markdown center-text">1. Portrait</div>
          <div class="center-text">Top Words: self, portrait</div>
          <p>Similar to findings from word frequency analysis, topic modeling reveals portraits as the most prevalent theme in art history.</p>
          <img src="data:image/jpg;base64,{encoded_image1}" alt="Example: Self-Portrait, Rembrandt" style="height:60%; max-height:300px; margin: auto; display: block;">
          <figcaption style="text-align:center; padding-top:0px; color:gray;">Example: <i>Self-Portrait</i>, Rembrandt</figcaption>
      </div>
      """,
      unsafe_allow_html=True
   )

with col2:
   st.markdown(
      f"""
      <div class="boxed-content">
          <div class="custom-markdown center-text">2. Water Bodies & Transportation</div>
          <div class="center-text">Top Words: beach, river, boat, lake, port</div>
          <p>The second most popular topic revolves around water body scenes, as well as activities that occur in these settings.</p>
          <img src="data:image/jpg;base64,{encoded_image2}" alt="Example: The Sea Port in Amsterdam, Claude Monet" style="width:100%; max-width:400px; margin: auto; display: block;">
          <figcaption style="text-align:center; padding-top:0px; color:gray;">Example: <i>The Sea Port in Amsterdam</i>, Claude Monet</figcaption>
      </div>
      """,
      unsafe_allow_html=True
   )

col1, col2 = st.columns(2)
    
with col1:
   st.markdown(
        f"""
        <div class="boxed-content">
            <div class="custom-markdown center-text">3. Nude Bathers</div>
            <div class="center-text">Top Words: nude, bather, female, bath, reclining</div>
            <p>Nudity and bathing scenes appear to be another common topic in artworks. These artworks depict undressed figures, possibly females, involved in bathing.</p>
            <img src="data:image/jpg;base64,{encoded_image3}" alt="Example: The Large Bathers, Paul Cézanne" style="width:100%; max-width:400px; margin: auto; display: block;">
            <figcaption style="text-align:center; padding-top:0px; color:gray;">Example: <i>The Large Bathers</i>, Paul Cézanne</figcaption>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
   st.markdown(
      f"""
      <div class="boxed-content">
          <div class="custom-markdown center-text">4. Animals</div>
          <div class="center-text">Top Words: horse, shepherd, dog, cow, deer</div>
          <p>This topic is related to various animals, such as horses and dogs, potentially referring to scenes from nature, farming, or hunting.</p>
          <img src="data:image/jpg;base64,{encoded_image4}" alt="Example: Horse Attacked by a Lion, Théodore Géricault" style="width:100%; max-width:400px; margin: auto; display: block;">
            <figcaption style="text-align:center; padding-top:0px; color:gray;">Example: <i>Horse Attacked by a Lion</i>, Théodore Géricault</figcaption>
      </div>
      """,
      unsafe_allow_html=True
   )

col1, col2 = st.columns(2)

with col1:
   st.markdown(
      f"""
      <div class="boxed-content">
          <div class="custom-markdown center-text">5. Seated or Resting Figures</div>
          <div class="center-text">Top Words: chair, sleeping, seated, sitting, woman</div>
          <p>This topic pertains to artworks depicting individuals in seated or resting positions.</p>
          <img src="data:image/jpg;base64,{encoded_image5}" alt="Example: Woman in Rocking Chair, Victor Borisov-Musatov" style="width:50%; max-width:400px; margin: auto; display: block;">
            <figcaption style="text-align:center; padding-top:0px; color:gray;">Example: <i> Woman in Rocking Chair</i>, Victor Borisov-Musatov</figcaption>
      </div>
      """,
      unsafe_allow_html=True
   )

with col2:
   st.markdown(
      f"""
      <div class="boxed-content">
          <div class="custom-markdown center-text">6. Entertainment</div>
          <div class="center-text">Top Words: player, ball, game, harlequin, card</div>
          <p>Artworks in this group depict various kinds of games (such as card games or ball games) and entertainment, including theatrical or carnival-like elements (i.e. "harlequin").</p>
          <img src="data:image/jpg;base64,{encoded_image6}" alt="Example: The Lute Player, Caravaggio" style="width:100%; max-width:400px; margin: auto; display: block;">
            <figcaption style="text-align:center; padding-top:0px; color:gray;">Example: <i> The Lute Player</i>, Caravaggio</figcaption>
      </div>
      """,
      unsafe_allow_html=True
   )

col1, col2 = st.columns(2)

with col1:
   st.markdown(
      f"""
      <div class="boxed-content">
          <div class="custom-markdown center-text">7. Fire</div>
          <div class="center-text">Top Words: fire, fireplace, smoking, burning, frieze</div>
          <p>This topic is about scenes or objects related to fire.</p>
          <img src="data:image/jpg;base64,{encoded_image7}" alt="Example: Fire at Night, Francisco Goya" style="width:50%; max-width:400px; margin: auto; display: block;">
            <figcaption style="text-align:center; padding-top:0px; color:gray;">Example: <i> Fire at Night</i>, Francisco Goya</figcaption>
      </div>
      """,
      unsafe_allow_html=True
   )

with col2:
   st.markdown(
      f"""
      <div class="boxed-content">
          <div class="custom-markdown center-text">8. Body Parts</div>
          <div class="center-text">Top Words: hand, arm, glove, leg, foot</div>
          <p>This topic revolves around different parts of the human body and items that can be worn on them.</p>
          <img src="data:image/jpg;base64,{encoded_image8}" alt="Example: Hands of an Apostle, Albrecht Durer" style="width:50%; max-width:400px; margin: auto; display: block;">
            <figcaption style="text-align:center; padding-top:0px; color:gray;">Example: <i> Hands of an Apostle</i>, Albrecht Durer</figcaption>
      </div>
      """,
      unsafe_allow_html=True
   )
   # st.image("https://static.streamlit.io/examples/owl.jpg")

st.subheader(""" Themes in Art in Different Periods of Time
             """)
# tab1, tab2, tab3, tab4, tab5= st.tabs(['Ancient Period(3050BCE - 0CE)',"Medieval Period (0 CE - 1500 CE)","Early Modern Period (1500 CE - 1760 CE)",\
#                                                    "Industrial Revolution (1760 CE - 1914 CE)","World War I to Present (1914 - 2023)"])


st.markdown("""
            <h2 style='color: #D85645; font-weight:600; font-size:1.5em;'>Ancient Period(3050BCE - 0CE)</h2>
            """, unsafe_allow_html=True)

st.markdown("""There are **346 artworks** in the dataset during this time period. Notably, many titles from this era are in Greek and Egyptian languages. \
            The predominant topic clusters identified include Ancient Greek art, containers, Egyptian art, drinking cups, and agriculture.""")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(
        """
        <div class="boxed-content-2">
            <div class="custom-markdown center-text">Ancient Greek Art</div>
            <div class="center-text">Top Words: greece, reconstruction, ancient, stela, fresco </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """
        <div class="boxed-content-2">
            <div class="custom-markdown center-text">Containers</div>
            <div class="center-text">Top Words: terracotta, jar, amphora, neck, bowl </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        """
        <div class="boxed-content-2">
            <div class="custom-markdown center-text">Egyptian Art</div>
            <div class="center-text">Top Words: tomb, nakht, nany, nebamun </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col4:
    st.markdown(
        """
        <div class="boxed-content-2">
            <div class="custom-markdown center-text">Drinking Cups</div>
            <div class="center-text">Top Words: water, hydria, wine, jar, terracotta </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col5:
    st.markdown(
        """
        <div class="boxed-content-2">
            <div class="custom-markdown center-text">Agriculture</div>
            <div class="center-text">Top Words: scarab, inscribed, named, maatkare, hatshepsut </div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.divider()


st.markdown("""
            <h2 style='color: #D85645; font-weight:600; font-size:1.5em;'>Medieval Period (0 CE - 1500 CE)</h2>
            """, unsafe_allow_html=True)

st.markdown("""There are **5806 artworks** in the dataset during this time period. This era features the rise of Christian iconography, figures,
            events and themes. \
            The topic clusters in artworks Gods/Angels, Annunciation, Crucifixion, and Christian Saints and Spiritual Figures.""")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        """
        <div class="boxed-content-2">
            <div class="custom-markdown center-text">Gods/Angels</div>
            <div class="center-text">Top Words: madonna, child, cherub, sleeping, seraph </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """
        <div class="boxed-content-2">
            <div class="custom-markdown center-text">Annunciation</div>
            <div class="center-text">Top Words: annunciation, annunciazionne, dyptic </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        """
        <div class="boxed-content-2">
            <div class="custom-markdown center-text">Crucifixion</div>
            <div class="center-text">Top Words: crucifixion, crucifix, pazzi, crucified, diptych </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col4:
    st.markdown(
        """
        <div class="boxed-content-2">
            <div class="custom-markdown center-text">Christian Saints and Spiritual Figures</div>
            <div class="center-text">Top Words: Stephen, Jerome, Sebastian, St </div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.divider()

st.markdown("""
            <h2 style='color: #D85645; font-weight:600; font-size:1.5em;'>Early Modern Period (1500 CE - 1760 CE)</h2>
            """, unsafe_allow_html=True)

st.markdown("""With **14,630 artworks** cataloged from this era, the early modern era is a prolific period of artistic production. \
            Even though this span of time is brief, it witnessed nearly three times the volume of artworks than the period \
            from 0 to 1500 CE. \
            This era signifies a notable shift in thematic focus from religious to more human-centric subjects.\
            
            Two major topic clusters emerge: portraits and religious architecture.\
            The 'portrait' cluster captures descriptions of various garments and notable facial expressions, such as "frowning". \
            The 'religious building' cluster encompasses terms like basilica, chapel, and ceiling.\
            
            Historically, this era aligns with the Baroque movement, which celebrates wealth, power and status throgh artworks, architecture and decorations. \
            Kings, princes, and popes began to prefer to see their own power and prestige emphasized through art than that of God.\
            In the topic modeling result, we can also witness this transition from purely religious themes to human focused themes.""")

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
        <div class="boxed-content-2">
            <div class="custom-markdown center-text">Portrait</div>
            <div class="center-text">Top Words: self, portrait, beret, gorget, frowning </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """
        <div class="boxed-content-2">
            <div class="custom-markdown center-text">Religious Buildings</div>
            <div class="center-text">Top Words: basilica, chapel, ceiling, sistine, church </div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.divider()


st.markdown("""
            <h2 style='color: #D85645; font-weight:600; font-size:1.5em;'>Industrial Revolution (1760 CE - 1914 CE)</h2>
            """, unsafe_allow_html=True)

st.markdown("""There are **38,025 artworks** during this time period, an even higher number than early modern period. While portrait\
            remians a prevalent theme, topic modeling reveals the emergence of landscape and horse themes during this time period.\
            
            Historical context provides insight into these trends. By the late 18th century, landscape paintings ascended in prominence,\
             a rise often attributed to the Romanticism movement. Concurrently, the 18th century witnessed a surge in equine art, \
            especially in England, which later spread across Europe. This century also marked the establishment of specialized schools\
             dedicated to animal and sporting art. Our topic modeling results resonate with these historical shifts, highlighting a notable\
             transition from religious and human-centric themes to a focus on natural landscapes and animals.""")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div class="boxed-content-2">
            <div class="custom-markdown center-text">Portrait</div>
            <div class="center-text">Top Words: self, portrait, redingote, yawning, andre </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """
        <div class="boxed-content-2">
            <div class="custom-markdown center-text">Landscape</div>
            <div class="center-text">Top Words: paysage, landscape, land, semmering, vexin </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        """
        <div class="boxed-content-2">
            <div class="custom-markdown center-text">Horse</div>
            <div class="center-text">Top Words: horse, horseman, horseback, stable, racehorse </div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.divider()

st.markdown("""
            <h2 style='color: #D85645; font-weight:600; font-size:1.5em;'>World War I - Present (1914 CE - 2023 CE)</h2>
            """, unsafe_allow_html=True)

st.markdown("""There are **65,416 artworks** during this time period. This time period shows four unique clusters: Animals, People, Abstract, Blue.\
            
            Notably, topic modeling for artworks from this time frame presents many complexities. \
            Comparing to other time periods, it required more efforts in fine tuning the model to form clear, coherent clusters.
            Moreover, interpreting results is complicated by ambiguous terms like "beginning," "come," and "heart." \
            I have carefully examined the characteristics of these artworks in order to create topic interpretations.\
            
            After looking into the data, I found that these intricacies could stem from the rise of contemporary art movements, \
            particularly abstract art.
            """)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        """
        <div class="boxed-content-2">
            <div class="custom-markdown center-text">Animals</div>
            <div class="center-text">Top Words: bird, hawk, oiseau, heron, sparrow </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """
        <div class="boxed-content-2">
            <div class="custom-markdown center-text">People</div>
            <div class="center-text">Top Words: ardoise, ariadne, elvis, emma </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        """
        <div class="boxed-content-2">
            <div class="custom-markdown center-text">Abstract</div>
            <div class="center-text">Top Words: composition, composicion, noter, duele, beginning </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col4:
    st.markdown(
        """
        <div class="boxed-content-2">
            <div class="custom-markdown center-text">Blue</div>
            <div class="center-text">Top Words: blue, blau, azul, bleue </div>
        </div>
        """,
        unsafe_allow_html=True
    )
  

st.divider()

st.header('Dimensionality Reduction + Clustering')

with st.expander("Read Methodology"):
  st.markdown("""
              <ul>
              <li> Calculated the embeddings of artwork titles using SentenceTransformer</li>
              <li> Used TSNE to reduce dimensions of the embeddings (reduced from 512 dimensions to 2 dimensions)</li>
              <li> Used KMeans and DBSCAN to discover clusters in the reduced space</li>
              <li> Observe produced clusters to find any shared characteristics within them (topics)
              </ul>""",
             unsafe_allow_html=True)
  st.text("""K-Means""")
  code = """kmeans = KMeans(n_clusters = 4, init='k-means++', n_init=10, max_iter=500) # Instantiating KMeans
kmeans = kmeans.fit(embeddings) # Fitting with inputs
labels = kmeans.predict(embeddings) # Predicting the clusters
source = pd.DataFrame(
    {
        'x': X_tsne[:, 0],
        'y': X_tsne[:, 1],
        'Artwork': corpus,
        'Topic': labels.astype(pd.Categorical)
    })
    
fig = px.scatter(
    source,
    x='x',
    y='y',
    color='Topic',
    hover_data=['Artwork', 'Topic'],
    title="Topics",
    labels={'x': 'not directly interpretable', 'y': 'not directly interpretable'},
)
fig.update_layout(width=600, height=600)
fig.show()"""
  st.code(code,language='python')
  
  st.text("""DBSCAN""")
  code = """
  db = DBSCAN(eps=1, min_samples=3) # Instantiate the model
  db.fit(embeddings) # Fit the model
  labels = db.labels_ # Get the labels
  source = pd.DataFrame(
    {
        'x': X_tsne[:, 0],
        'y': X_tsne[:, 1],
        'Artwork': corpus,
        'Topic': labels.astype(pd.Categorical)
    })
    
fig = px.scatter(
    source,
    x='x',
    y='y',
    color='Topic',
    hover_data=['Artwork', 'Topic'],
    title="Topics",
    labels={'x': 'not directly interpretable', 'y': 'not directly interpretable'},
)
fig.update_layout(width=600, height=600)
fig.show()"""
  st.code(code,language='python')

st.subheader('K-Means with 4 Clusters')

col1, col2 = st.columns([3,2])

with col1:
    with open('plots/kmeans_4.html', 'r') as f:
        content = f.read()
        # centered_content = f'<div style="display: flex; justify-content: center;">{content}</div>'
        st.components.v1.html(content, height=600)

with col2:
    st.markdown("""
                <div style="background-color: rgba(238,99,82,0.2); margin-top: 140px; padding: 25px; border-radius: 10px;">
                By setting the number of clusters to four, this approach shows Cluster 0 contains "untitled" artworks, \
                while the green group contains "illustration" and "portrait". The themes of the red and blue groups remain hard to interpret. \
                To gain clearer insights, we will increase the number of clusters.
                </div>
                """, unsafe_allow_html=True)

st.subheader('K-Means with 8 Clusters')

col1, col2 = st.columns([3,2])

with col1:
    with open('plots/kmeans_8.html', 'r') as f:
        content = f.read()
        # centered_content = f'<div style="display: flex; justify-content: center;">{content}</div>'
        st.components.v1.html(content, height=600)

with col2:
    st.markdown("""
                <div style="background-color: rgba(238,99,82,0.2); margin-top: 140px; padding: 25px; border-radius: 10px;">
                We now set the number of clusters to eight. After examining the resulting clusters, we can see that:

                - **Topic 3**: Appears to be centered around themes of femininity, encompassing terms like "woman," "girl," "miss," and "virgin."
                - **Topic 0**: Primarily focused on "portrait" and "illustration."
                - **Topic 5**: Predominantly "untitled" artworks.

                However, discerning shared characteristics in the remaining clusters proves challenging. While this exploration has been enlightening, K-Means may not provide the most comprehensive insights into the prevailing themes in art.
                </div>
                """, unsafe_allow_html=True)

st.subheader('DBSCAN')

col1, col2 = st.columns([3,2])

with col1:
    with open('plots/dbscan.html', 'r') as f:
        content = f.read()
        st.components.v1.html(content, height=600)

with col2:
    st.markdown("""
                <div style="background-color: rgba(238,99,82,0.2); margin-top: 140px; padding: 25px; border-radius: 10px;">
                We now explore DBSCAN, a density-based clustering technique, which has the advantage of not requiring us to specify the number of clusters.
                
                However, DBSCAN did not appear to perform better than K-Means. It yielded two clusters, with one consisting of artworks with exceptionally long titles. \
                No shared topics within these clusters were observed.
                </div>
                """, unsafe_allow_html=True)
    


st.header('Conclusion')

st.markdown("""
            <div style="background-color: rgba(238,99,82,0.2); padding: 25px; border-radius: 10px;">
            In this project, we delved into the thematic analysis of artworks over time, utilizing a triad of methodologies: word frequency analysis, topic modeling, and a combination of dimensionality reduction and clustering. Each technique illuminated unique aspects of our subject matter.

            1. **Word Frequency Analysis**: This approach shed light on the prevalence of specific words within artwork titles across different periods, offering a granular view of thematic evolution.

            2. **Topic Modeling with BERT**: Leveraging BERT's profound contextual capabilities, we gained insights into broader thematic shifts over various periods. This technique revealed a fascinating transition in prevalent themes, ranging from agricultural and religious motifs to more contemporary focuses like cultural icons, including Elvis.

            3. **Dimensionality Reduction and Clustering**: This experimental approach enabled us to visualize the condensation of artwork title embeddings into a two-dimensional space and investigate the clusters. Our findings corroborated the dominance of themes like "untitled," "portrait," and "illustration." Notably, a recurrent motif of femininity surfaced, aligning with our word frequency analysis that identified "woman" as a historically prominent term.

            Together, these methods revealed the evolution of themes in artworks. Moving forward, I plan to explore computer vision and gather more detailed artwork descriptions to enhance the depth and accuracy of my insights.
            </div>
            """, unsafe_allow_html=True)

