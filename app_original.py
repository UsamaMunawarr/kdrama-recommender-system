import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
data = pd.read_csv('kdrama_list.csv')

# Preprocessing: Combine 'Genre' and 'Tags'

################################
# write a function on which we will select 1st two genre in each Genre convert it into a string
def genre(x):
    return x.split(',')[0:2]
# apply the function on Genre column
data['Genre'] = data['Genre'].apply(genre)
# convert Genre into string
data['Genre'] = data['Genre'].apply(lambda x: ', '.join(x))
################################################
#data = data.dropna(subset=['Tags', 'Sinopsis'])  # Drop missing values
data = data.dropna(subset=['Tags'])  # Drop missing values
#data['combined_features'] = data['Genre'].str.lower() + ' ' + data['Tags'].str.lower() + ' ' + data['Sinopsis'].str.lower()
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
        sim_scores = sim_scores[1:5]  # Skip the first one because it's the same drama

        # Get the drama indices
        drama_indices = [i[0] for i in sim_scores]  # This was missing

        # Return the top 5 most similar dramas along with their images
        return data[['Name', 'img url']].iloc[drama_indices]
    except IndexError:
        return None

# Streamlit app interface
st.title("K-Drama Recommender")

# Dropdown menu for drama selection
drama_names = data['Name'].tolist()
selected_drama = st.selectbox("Select a K-Drama you like:", options=drama_names)

# Display recommendations
if st.button('Get Recommendations'):
    if selected_drama:
        recommendations = get_recommendations(selected_drama)
        if recommendations is not None:
            st.write(f"Top 5 recommendations for '{selected_drama}':")

            # Create columns dynamically based on the number of recommendations
            cols = st.columns(len(recommendations))

            # Display each recommended drama and its image in a row
            for i, (idx, row) in enumerate(recommendations.iterrows()):
                with cols[i]:
                    st.write(f"{i+1}. {row['Name']}")
                    if pd.notna(row['img url']):  # Check if the URL is not NaN
                        st.image(row['img url'], width=150)  # Display the image
        else:
            st.write("K-Drama not found. Please check the name and try again.")
    else:
        st.write("Please select a K-Drama name.")
