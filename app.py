import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#####################################
# Set Streamlit to full screen mode##
#####################################
st.set_page_config(
    page_title="K-Drama Recommender App",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

########
# Intro#
########
selected = option_menu(
    menu_title=None,
    options=["Home", "App", "Contact"],
    icons=["house", "film", "envelope"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "25px"},
        "nav-link": {
            "font-size": "25px",
            "text-align": "left",
            "margin": "0px",
            "--hover-color": "#eee",
        },
        "nav-link-selected": {"background-color": "green"},
    },
)

##############
# Home Page ##
##############
if selected == "Home":
    st.title("Welcome to the K-Drama Recommender! 🎬")
    st.image("k1.png", use_column_width=True)
    st.write("""
        Discover your next favorite K-Drama! This app helps you find similar dramas based on what you like. 
        Head to the 'App' section to choose a drama and get recommendations.
    """)

################
# App Page     #
################
elif selected == "App":
    st.title("K-Drama Recommender App 🎥")

    # Load data
    data = pd.read_csv('kdrama_list.csv')

    # Preprocessing: Combine 'Genre' and 'Tags'
    def genre(x):
        return x.split(',')[0:2]

    # Apply the function to the Genre column
    data['Genre'] = data['Genre'].apply(genre)

    # Convert Genre into a string
    data['Genre'] = data['Genre'].apply(lambda x: ', '.join(x))

    # Drop missing values in the 'Tags' column
    data = data.dropna(subset=['Tags'])

    # Create a combined features column for recommendations
    data['combined_features'] = data['Genre'].str.lower() + ' ' + data['Tags'].str.lower()

    # Convert combined features into a matrix of token counts
    vectorizer = CountVectorizer(tokenizer=lambda x: x.split(', '))
    features_matrix = vectorizer.fit_transform(data['combined_features'])

    # Compute the cosine similarity matrix
    cosine_sim = cosine_similarity(features_matrix)

    # Function to get top 5 recommendations
    def get_recommendations(drama_name, cosine_sim=cosine_sim):
        try:
            # Get the index of the K-drama that matches the title
            idx = data[data['Name'].str.lower() == drama_name.lower()].index[0]

            # Get the similarity scores for this drama with all others
            sim_scores = list(enumerate(cosine_sim[idx]))

            # Sort the dramas based on similarity scores (in descending order)
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

            # Get the indices of the 5 most similar dramas (excluding itself)
            sim_scores = sim_scores[1:5]

            # Get the drama indices
            drama_indices = [i[0] for i in sim_scores]

            # Return the top 5 most similar dramas along with their images
            return data[['Name', 'img url']].iloc[drama_indices]
        except IndexError:
            return None

    # Dropdown menu for drama selection
    drama_names = data['Name'].tolist()
    selected_drama = st.selectbox("Select a K-Drama you like:", options=drama_names)

    # Display recommendations
    if st.button('Get Recommendations'):
        if selected_drama:
            recommendations = get_recommendations(selected_drama)
            if recommendations is not None:
                st.write(f"Top 5 recommendations for '{selected_drama}':")

                # Display each recommended drama and its image in a row
                cols = st.columns(len(recommendations))
                for i, (idx, row) in enumerate(recommendations.iterrows()):
                    with cols[i]:
                        st.write(f"{i+1}. {row['Name']}")
                        if pd.notna(row['img url']):  # Check if the URL is valid
                            st.image(row['img url'], width=150)
            else:
                st.write("K-Drama not found. Please check the name and try again.")
        else:
            st.write("Please select a K-Drama name.")

#################
# Contact Page  #
#################
elif selected == "Contact":
    #st.title(f"You have selected {selected}")

    
    
    ### About the author
    st.write("##### About the author:")
    
    ### Author name
    st.write("<p style='color:blue; font-size: 50px; font-weight: bold;'>Usama Munawar</p>", unsafe_allow_html=True)
    
    ### Connect on social media
    st.write("##### Connect with me on social media")
    
    ### Add social media links
    ### URLs for images
    linkedin_url = "https://img.icons8.com/color/48/000000/linkedin.png"
    github_url = "https://img.icons8.com/fluent/48/000000/github.png"
    youtube_url = "https://img.icons8.com/?size=50&id=19318&format=png"
    twitter_url = "https://img.icons8.com/color/48/000000/twitter.png"
    facebook_url = "https://img.icons8.com/color/48/000000/facebook-new.png"
    
    ### Redirect URLs
    linkedin_redirect_url = "https://www.linkedin.com/in/abu--usama"
    github_redirect_url = "https://github.com/UsamaMunawarr"
    youtube_redirect_url ="https://www.youtube.com/@CodeBaseStats"
    twitter_redirect_url = "https://twitter.com/Usama__Munawar?t=Wk-zJ88ybkEhYJpWMbMheg&s=09"
    facebook_redirect_url = "https://www.facebook.com/profile.php?id=100005320726463&mibextid=9R9pXO"
    
    ### Add links to images
    st.markdown(f'<a href="{github_redirect_url}"><img src="{github_url}" width="60" height="60"></a>'
                f'<a href="{linkedin_redirect_url}"><img src="{linkedin_url}" width="60" height="60"></a>'
                f'<a href="{youtube_redirect_url}"><img src="{youtube_url}" width="60" height="60"></a>'
                f'<a href="{twitter_redirect_url}"><img src="{twitter_url}" width="60" height="60"></a>'
                f'<a href="{facebook_redirect_url}"><img src="{facebook_url}" width="60" height="60"></a>', unsafe_allow_html=True)

###################
# Thank you footer#
###################
st.write("<p style='color:green; font-size: 30px; font-weight: bold;'>Thank you for using this app, share it with your friends! 😊</p>", unsafe_allow_html=True)
