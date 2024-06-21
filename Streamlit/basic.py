import streamlit as st
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 



#Load sample dataset or utilize your own dataset 
@st.cache

def load_data():
    return sns.load_dataset('iris')

#Main function to define the Streamlit app 
def main():
    st.title('Exploratory Data analysis dashboard (EDA')
    
    #Load dataset 
    data=load_data()
    
    #Display the dataset 
    st.subheader("Dataset")
    st.write(data)
    
    #Interactive widgets for filtering data
    species=st.sidebar.multiselect(
    'Select Species',data['species'].unique()
    )
    sepal_length_range = st.sidebar.slider('Select Sepal Length Range', float(data['sepal_length'].min()), float(data['sepal_length'].max()), (float(data['sepal_length'].min()), float(data['sepal_length'].max())))
      
     # Filter data based on user inputs
    filtered_data = data[(data['species'].isin(species)) & 
                         (data['sepal_length'] >= sepal_length_range[0]) & 
                         (data['sepal_length'] <= sepal_length_range[1])]
    
    # Display filtered data
    st.subheader('Filtered Data')
    st.write(filtered_data)
    
    # Visualization section
    st.subheader('Data Visualization')
    
    # Histogram of Sepal Length
    st.write('## Histogram of Sepal Length')
    plt.figure(figsize=(8, 6))
    sns.histplot(data=filtered_data, x='sepal_length', bins=20, kde=True)
    st.pyplot()
    
    # Scatter plot of Sepal Length vs Sepal Width
    st.write('## Scatter Plot of Sepal Length vs Sepal Width')
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=filtered_data, x='sepal_length', y='sepal_width', hue='species')
    st.pyplot()

# Run the app
if __name__ == '__main__':
    main() 