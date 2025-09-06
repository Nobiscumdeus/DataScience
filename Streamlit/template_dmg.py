import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="Medical Student Analytics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("🏥 Medical Student Performance Analytics - Debug Version")
    
    # Sidebar
    st.sidebar.header("📁 Data Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Upload Excel file with student results",
        type=['xlsx', 'xls']
    )
    
    if uploaded_file is not None:
        # Read the file
        df = pd.read_excel(uploaded_file)
        
        # Show raw data first
        st.header("🔍 Raw Data Analysis")
        st.subheader("First 5 rows of your data:")
        st.dataframe(df.head())
        
        st.subheader("Column Information:")
        col_info = []
        for i, col in enumerate(df.columns):
            col_type = str(df[col].dtype)
            sample_vals = df[col].head(3).tolist()
            col_info.append({
                'Index': i,
                'Column Name': col,
                'Data Type': col_type,
                'Sample Values': str(sample_vals)
            })
        
        col_df = pd.DataFrame(col_info)
        st.dataframe(col_df)
        
        # Let user manually select columns
        st.header("📊 Manual Column Selection")
        
        name_col = st.selectbox("Select STUDENT NAME column:", df.columns)
        matric_col = st.selectbox("Select MATRIC NUMBER column:", df.columns, index=1 if len(df.columns) > 1 else 0)
        
        # Get remaining columns (excluding name and matric)
        remaining_cols = [col for col in df.columns if col not in [name_col, matric_col]]
        
        st.subheader("Select Assessment Columns:")
        physio_cols = st.multiselect("Physio columns:", remaining_cols)
        anatomy_cols = st.multiselect("Anatomy columns:", remaining_cols)
        bich_cols = st.multiselect("Biochemistry columns:", remaining_cols)
        
        if st.button("Analyze Data"):
            if physio_cols or anatomy_cols or bich_cols:
                
                # Convert assessment columns to numeric
                all_assessment_cols = physio_cols + anatomy_cols + bich_cols
                for col in all_assessment_cols:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Calculate averages
                if physio_cols:
                    df['Physio_Avg'] = df[physio_cols].mean(axis=1)
                if anatomy_cols:
                    df['Anatomy_Avg'] = df[anatomy_cols].mean(axis=1)
                if bich_cols:
                    df['Bich_Avg'] = df[bich_cols].mean(axis=1)
                
                # Calculate overall average
                avg_cols = []
                if physio_cols:
                    avg_cols.append('Physio_Avg')
                if anatomy_cols:
                    avg_cols.append('Anatomy_Avg')
                if bich_cols:
                    avg_cols.append('Bich_Avg')
                
                if avg_cols:
                    df['Overall_Avg'] = df[avg_cols].mean(axis=1)
                
                # Show calculated averages
                st.header("📈 Calculated Results")
                
                display_cols = [name_col]
                if physio_cols:
                    display_cols.extend(physio_cols + ['Physio_Avg'])
                if anatomy_cols:
                    display_cols.extend(anatomy_cols + ['Anatomy_Avg'])
                if bich_cols:
                    display_cols.extend(bich_cols + ['Bich_Avg'])
                if avg_cols:
                    display_cols.append('Overall_Avg')
                
                result_df = df[display_cols].round(2)
                st.dataframe(result_df)
                
                # Risk Analysis
                if 'Overall_Avg' in df.columns:
                    st.header("🚨 Risk Analysis")
                    
                    # Define risk levels
                    def get_risk_level(score):
                        if pd.isna(score):
                            return "High Risk"
                        elif score < 35:
                            return "Critical Risk"
                        elif score < 50:
                            return "High Risk"
                        elif score < 65:
                            return "Medium Risk"
                        else:
                            return "Low Risk"
                    
                    df['Risk_Level'] = df['Overall_Avg'].apply(get_risk_level)
                    
                    # Show risk distribution
                    risk_summary = df['Risk_Level'].value_counts()
                    st.subheader("Risk Distribution:")
                    for risk, count in risk_summary.items():
                        st.write(f"**{risk}**: {count} students")
                    
                    # Show students by risk
                    st.subheader("Students by Risk Level:")
                    
                    for risk_level in ['Critical Risk', 'High Risk', 'Medium Risk', 'Low Risk']:
                        risk_students = df[df['Risk_Level'] == risk_level]
                        if not risk_students.empty:
                            st.write(f"**{risk_level} ({len(risk_students)} students):**")
                            for _, student in risk_students.iterrows():
                                score = student['Overall_Avg']
                                if pd.isna(score):
                                    score_text = "No data"
                                else:
                                    score_text = f"{score:.1f}%"
                                st.write(f"- {student[name_col]}: {score_text}")
                    
                    # Statistics
                    st.header("📊 Statistics")
                    valid_scores = df['Overall_Avg'].dropna()
                    if len(valid_scores) > 0:
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Average Score", f"{valid_scores.mean():.1f}%")
                        with col2:
                            st.metric("Highest Score", f"{valid_scores.max():.1f}%")
                        with col3:
                            st.metric("Lowest Score", f"{valid_scores.min():.1f}%")
                        with col4:
                            st.metric("Students with Data", len(valid_scores))
                    
                    # Charts
                    if len(valid_scores) > 0:
                        st.header("📊 Visualizations")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Risk distribution pie chart
                            fig_pie = px.pie(
                                values=risk_summary.values,
                                names=risk_summary.index,
                                title="Risk Distribution"
                            )
                            st.plotly_chart(fig_pie, use_container_width=True)
                        
                        with col2:
                            # Score distribution histogram
                            fig_hist = px.histogram(
                                x=valid_scores,
                                title="Score Distribution",
                                labels={'x': 'Score (%)', 'y': 'Number of Students'}
                            )
                            st.plotly_chart(fig_hist, use_container_width=True)
            
            else:
                st.warning("Please select at least one assessment column to analyze.")
    
    else:
        st.info("👆 Please upload an Excel file to begin analysis")
        
        # Show expected format
        st.subheader("📋 Expected Excel Format:")
        sample_data = {
            'FULL NAMES': ['John Doe', 'Jane Smith', 'Bob Johnson'],
            'MATRIC NO': ['MED001', 'MED002', 'MED003'],
            'Physio CA1': [75, 68, 82],
            'Physio CA2': [78, 65, 85],
            'Anatomy CA1': [70, 72, 78],
            'Anatomy CA2': [73, 69, 80],
            'Bich CA1': [65, 70, 75],
            'Bich CA2': [68, 72, 77]
        }
        st.dataframe(pd.DataFrame(sample_data))

if __name__ == "__main__":
    main()