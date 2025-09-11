import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Interactive Performance Insights for Medical Students",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .risk-high { color: #ff4444; font-weight: bold; }
    .risk-medium { color: #ffaa00; font-weight: bold; }
    .risk-low { color: #00aa44; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

def load_and_process_data(uploaded_file):
    """Load Excel file and process the data dynamically"""
    try:
        # Read Excel file
        df = pd.read_excel(uploaded_file)
        
        # Clean column names - remove extra spaces
        df.columns = df.columns.str.strip()
        
        # Convert all numeric columns properly
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='ignore')
        
        return df
    
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def calculate_subject_averages(df):
    """Calculate averages for each subject grouping - FIXED VERSION"""
    subject_groups = {}
    
    # More flexible subject detection
    physio_cols = [col for col in df.columns if 'physio' in str(col).lower()]
    anatomy_cols = [col for col in df.columns if 'anatomy' in str(col).lower() or 'anat' in str(col).lower()]
    bich_cols = [col for col in df.columns if any(term in str(col).lower() for term in ['bich', 'biochem', 'biochemistry'])]
    
    st.sidebar.write("🔍 Subject Detection:")
    st.sidebar.write(f"✅ Physio columns: {physio_cols}")
    st.sidebar.write(f"✅ Anatomy columns: {anatomy_cols}")
    st.sidebar.write(f"✅ Bich columns: {bich_cols}")
    
    # Calculate averages using the WORKING method from debug version
    if physio_cols:
        df['physio_average'] = df[physio_cols].mean(axis=1)
        subject_groups['physio'] = physio_cols
        st.sidebar.write(f"Physio avg range: {df['physio_average'].min():.1f} - {df['physio_average'].max():.1f}")
    
    if anatomy_cols:
        df['anatomy_average'] = df[anatomy_cols].mean(axis=1)
        subject_groups['anatomy'] = anatomy_cols
        st.sidebar.write(f"Anatomy avg range: {df['anatomy_average'].min():.1f} - {df['anatomy_average'].max():.1f}")
    
    if bich_cols:
        df['bich_average'] = df[bich_cols].mean(axis=1)
        subject_groups['bich'] = bich_cols
        st.sidebar.write(f"Bich avg range: {df['bich_average'].min():.1f} - {df['bich_average'].max():.1f}")
    
    return df, subject_groups

def identify_risk_students(df):
    """Identify students at risk based on performance"""
    risk_analysis = []
    
    # Get average columns
    avg_cols = [col for col in df.columns if '_average' in col]
    
    # Get the first column (should be student names)
    name_col = df.columns[0]
    
    for idx in range(len(df)):
        student_name = df.iloc[idx][name_col]
        
        # Calculate overall average using the WORKING method
        if avg_cols:
            overall_avg = df.iloc[idx][avg_cols].mean()
        else:
            # Fallback: use all numeric columns except first two
            numeric_cols = df.select_dtypes(include=[np.number]).columns[2:]  # Skip name and matric
            if len(numeric_cols) > 0:
                overall_avg = df.iloc[idx][numeric_cols].mean()
            else:
                overall_avg = 0
        
        # Risk categorization - Medical school appropriate (pass = 50%)
        if pd.isna(overall_avg):
            risk_level = "High Risk"
            risk_color = "risk-high"
            overall_avg = 0
        elif overall_avg < 35:
            risk_level = "Critical Risk"
            risk_color = "risk-high"
        elif overall_avg < 50:
            risk_level = "High Risk"
            risk_color = "risk-high"
        elif overall_avg < 65:
            risk_level = "Medium Risk" 
            risk_color = "risk-medium"
        else:
            risk_level = "Low Risk"
            risk_color = "risk-low"
        
        # Check for declining performance
        declining_subjects = []
        for subject in ['physio', 'anatomy', 'bich']:
            subject_cols = [col for col in df.columns 
                          if subject.lower() in str(col).lower() and 'average' not in str(col).lower()]
            
            # Remove name/matric columns
            subject_cols = [col for col in subject_cols 
                          if not any(pattern in str(col).lower() for pattern in ['name', 'matric', 'id'])]
            
            if len(subject_cols) >= 2:
                scores = []
                for col in subject_cols:
                    val = df.iloc[idx][col]
                    if pd.notna(val):
                        scores.append(val)
                
                if len(scores) >= 2 and scores[-1] < scores[-2]:
                    declining_subjects.append(subject)
        
        # Count missing assessments (only NaN, not zeros)
        missing_count = 0
        for col in df.columns[2:]:  # Skip name and matric
            val = df.iloc[idx][col]
            if pd.isna(val):
                missing_count += 1
        
        risk_analysis.append({
            'Student': student_name,
            'Overall_Average': round(overall_avg, 1) if not pd.isna(overall_avg) else 0,
            'Risk_Level': risk_level,
            'Risk_Color': risk_color,
            'Declining_Subjects': ', '.join(declining_subjects),
            'Missing_Assessments': missing_count
        })
    
    return pd.DataFrame(risk_analysis)

def generate_peer_matches(df, risk_df):
    """Generate peer tutoring matches with balanced tutor distribution"""
    matches = []
    
    # Subject-wise matching
    for subject in ['physio', 'anatomy', 'bich']:
        avg_col = f'{subject}_average'
        if avg_col in df.columns:
            # Get students with scores for this subject
            subject_data = df[[df.columns[0], avg_col]].dropna()
            if len(subject_data) < 2:
                continue
            
            # Sort by score
            subject_data = subject_data.sort_values(avg_col, ascending=False)
            
            # Identify struggling students (bottom 30% or score < 50)
            threshold_30_percent = int(len(subject_data) * 0.3)
            struggling_by_rank = subject_data.tail(max(threshold_30_percent, 3))  # At least 3
            struggling_by_score = subject_data[subject_data[avg_col] < 50]
            
            # Combine and deduplicate struggling students - BUT filter out decent performers
            struggling_names = set(struggling_by_rank.iloc[:, 0].tolist() + struggling_by_score.iloc[:, 0].tolist())
            struggling_candidates = subject_data[subject_data.iloc[:, 0].isin(struggling_names)]
            
            # IMPROVED LOGIC: Only keep students who ACTUALLY need help (score < 60)
            # Someone with 63% doesn't need tutoring even if they're in bottom 30%
            struggling = struggling_candidates[struggling_candidates[avg_col] < 60]
            
            # Dynamic tutor selection based on need
            num_struggling = len(struggling)
            if num_struggling <= 3:
                num_tutors = 2  # 2 tutors for small groups
            elif num_struggling <= 6:
                num_tutors = 3  # 3 tutors for medium groups  
            elif num_struggling <= 9:
                num_tutors = 4  # 4 tutors for larger groups
            else:
                num_tutors = max(4, num_struggling // 3)  # Scale up for very large groups
            
            # Get top performers (must score >75 and significantly better than struggling students)
            potential_tutors = subject_data[subject_data[avg_col] > 75].head(num_tutors * 2)  # Get extra options
            qualified_tutors = []
            
            for _, tutor in potential_tutors.iterrows():
                # Check if tutor is significantly better than average struggling student
                avg_struggling_score = struggling[avg_col].mean()
                if tutor[avg_col] > avg_struggling_score + 20:  # 15 point minimum gap
                    qualified_tutors.append(tutor)
                    if len(qualified_tutors) >= num_tutors:
                        break
            
            # If we don't have enough qualified tutors, lower the bar slightly
            if len(qualified_tutors) < max(2, num_struggling // 4):
                additional_tutors = subject_data[
                    (subject_data[avg_col] > 70) & 
                    (~subject_data.iloc[:, 0].isin([t.iloc[0] for t in qualified_tutors]))
                ].head(num_tutors - len(qualified_tutors))
                qualified_tutors.extend([tutor for _, tutor in additional_tutors.iterrows()])
            
            # Create balanced matches
            if qualified_tutors:
                tutor_assignments = {tutor.iloc[0]: 0 for tutor in qualified_tutors}  # Track tutor load
                
                # Sort struggling students by score (worst first) for fair distribution
                struggling_sorted = struggling.sort_values(avg_col, ascending=True)
                
                for _, struggling_student in struggling_sorted.iterrows():
                    # Find tutor with least assignments
                    least_busy_tutor = min(tutor_assignments.keys(), key=lambda x: tutor_assignments[x])
                    
                    # Get tutor data
                    tutor_data = next(t for t in qualified_tutors if t.iloc[0] == least_busy_tutor)
                    
                    # Ensure different students and significant gap
                    if (tutor_data.iloc[0] != struggling_student.iloc[0] and 
                        tutor_data[avg_col] > struggling_student[avg_col] + 5):
                        
                        matches.append({
                            'Subject': subject.title(),
                            'Tutor': tutor_data.iloc[0],
                            'Tutor_Score': round(tutor_data[avg_col], 1),
                            'Student': struggling_student.iloc[0],
                            'Student_Score': round(struggling_student[avg_col], 1),
                            'Score_Gap': round(tutor_data[avg_col] - struggling_student[avg_col], 1),
                            'Tutor_Load': tutor_assignments[least_busy_tutor] + 1
                        })
                        
                        # Update tutor load
                        tutor_assignments[least_busy_tutor] += 1
    
    return pd.DataFrame(matches).drop_duplicates() if matches else pd.DataFrame()

def main():
    st.title("🏥 Interactive Performance Insights for Medical Students")
    st.markdown("*Analyze CA results and optimize reading group study sessions*")
    
    # Sidebar
    st.sidebar.header("📁 Data Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Upload Excel file with student results",
        type=['xlsx', 'xls'],
        help="Expected columns: Name, Matric No, Subject CAs (e.g., Physio CA1, Anatomy CA2, etc.)"
    )
    
    if uploaded_file is not None:
        # Load and process data
        with st.spinner("Processing data..."):
            df = load_and_process_data(uploaded_file)
        
        if df is not None:
            st.sidebar.success(f"✅ Loaded {len(df)} students")
            
            # Show data preview
            st.sidebar.subheader("📋 Data Preview:")
            st.sidebar.dataframe(df.head(2).reset_index(drop=True), use_container_width=True,hide_index=True)
            
            # Process data
            df, subject_groups = calculate_subject_averages(df)
            risk_df = identify_risk_students(df)
            
            # Check if any averages were calculated
            avg_cols = [col for col in df.columns if '_average' in col]
            if avg_cols:
                st.sidebar.info(f"📊 Calculated averages for {len(avg_cols)} subjects")
            else:
                st.sidebar.warning("⚠️ No subject averages calculated - check column names")
            
            peer_matches = generate_peer_matches(df, risk_df)
            
            # Main dashboard
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                critical_high_count = len(risk_df[risk_df['Risk_Level'].isin(['Critical Risk', 'High Risk'])])
                st.metric("🚨 High/Critical Risk", critical_high_count)
            
            with col2:
                medium_risk_count = len(risk_df[risk_df['Risk_Level'] == 'Medium Risk'])
                st.metric("⚠️ Medium Risk Students", medium_risk_count)
            
            with col3:
                avg_score = risk_df['Overall_Average'].mean()
                st.metric("📊 Group Average", f"{avg_score:.1f}%")
            
            with col4:
                total_missing = risk_df['Missing_Assessments'].sum()
                st.metric("❓ Missing Assessments", total_missing)
            
            # Performance Overview
            st.header("📈 Performance Overview")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Risk distribution pie chart
                risk_counts = risk_df['Risk_Level'].value_counts()
                fig_pie = px.pie(
                    values=risk_counts.values,
                    names=risk_counts.index,
                    title="Student Risk Distribution",
                    color_discrete_map={
                        'Critical Risk': '#cc0000',
                        'High Risk': '#ff4444',
                        'Medium Risk': '#ffaa00', 
                        'Low Risk': '#00aa44'
                    }
                )
                st.plotly_chart(fig_pie, use_container_width=True,hide_index=True)
            
            with col2:
                # Subject averages bar chart
                subject_avgs = {}
                for subject in ['physio', 'anatomy', 'bich']:
                    avg_col = f'{subject}_average'
                    if avg_col in df.columns:
                        avg_val = df[avg_col].mean()
                        if not np.isnan(avg_val):
                            subject_avgs[subject.title()] = avg_val
                
                if subject_avgs:
                    fig_bar = px.bar(
                        x=list(subject_avgs.keys()),
                        y=list(subject_avgs.values()),
                        title="Average Scores by Subject",
                        labels={'x': 'Subject', 'y': 'Average Score (%)'},
                        text=[f"{v:.1f}%" for v in subject_avgs.values()]
                    )
                    fig_bar.update_traces(marker_color='lightblue', textposition='outside')
                    st.plotly_chart(fig_bar, use_container_width=True,hide_index=True)
                else:
                    st.info("No subject averages available for chart")
            
            # Student Performance Table
            st.header("👥 Student Performance Analysis")
            
            # Filters and Sorting
            col1, col2, col3 = st.columns(3)
            with col1:
                risk_filter = st.multiselect(
                    "Filter by Risk Level",
                    options=['Critical Risk', 'High Risk', 'Medium Risk', 'Low Risk'],
                    default=['Critical Risk', 'High Risk', 'Medium Risk', 'Low Risk']
                )
            
            with col2:
                score_threshold = st.slider("Minimum Score Threshold", 0, 100, 0)
            
            with col3:
                sort_option = st.selectbox(
                    "Sort by:",
                    options=['Score (High to Low)', 'Score (Low to High)', 'Name (A-Z)', 'Risk Level'],
                    index=0  # Default to Score (High to Low)
                )
            
            # Apply filters
            filtered_risk = risk_df[
                (risk_df['Risk_Level'].isin(risk_filter)) &
                (risk_df['Overall_Average'] >= score_threshold)
            ]
            
            # Apply sorting
            if sort_option == 'Score (High to Low)':
                filtered_risk = filtered_risk.sort_values('Overall_Average', ascending=False)
            elif sort_option == 'Score (Low to High)':
                filtered_risk = filtered_risk.sort_values('Overall_Average', ascending=True)
            elif sort_option == 'Name (A-Z)':
                filtered_risk = filtered_risk.sort_values('Student', ascending=True)
            elif sort_option == 'Risk Level':
                # Custom sort: Critical -> High -> Medium -> Low
                risk_order = {'Critical Risk': 0, 'High Risk': 1, 'Medium Risk': 2, 'Low Risk': 3}
                filtered_risk['risk_sort'] = filtered_risk['Risk_Level'].map(risk_order)
                filtered_risk = filtered_risk.sort_values(['risk_sort', 'Overall_Average'], ascending=[True, False])
                filtered_risk = filtered_risk.drop('risk_sort', axis=1)
            
            # Display filtered and sorted results
            st.dataframe(
                filtered_risk[['Student', 'Overall_Average', 'Risk_Level', 
                             'Declining_Subjects', 'Missing_Assessments']],
                use_container_width=True,hide_index=True
            )
            
            # Peer Tutoring Recommendations
            st.header("🤝 Peer Tutoring Matches")
            if not peer_matches.empty:
                # Show tutor workload summary
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.dataframe(peer_matches, use_container_width=True,hide_index=True)
                
                with col2:
                    st.subheader("📊 Tutor Workload")
                    tutor_load = peer_matches.groupby(['Tutor', 'Subject']).size().reset_index(name='Students_Assigned')
                    total_load = tutor_load.groupby('Tutor')['Students_Assigned'].sum().reset_index()
                    total_load = total_load.sort_values('Students_Assigned', ascending=False)
                    
                    for _, row in total_load.iterrows():
                        tutor_name = row['Tutor']
                        load = row['Students_Assigned']
                        if load > 3:
                            st.error(f"⚠️ {tutor_name}: {load} students")
                        elif load > 2:
                            st.warning(f"📚 {tutor_name}: {load} students")
                        else:
                            st.success(f"✅ {tutor_name}: {load} students")
                
                # Enhanced download options
                st.subheader("💾 Download Options")
                col_download1, col_download2, col_download3 = st.columns(3)
                
                with col_download1:
                    csv_data = peer_matches.to_csv(index=False)
                    st.download_button(
                        label="📄 Download CSV",
                        data=csv_data,
                        file_name='peer_tutoring_matches.csv',
                        mime='text/csv'
                    )
                
                with col_download2:
                    # Convert to Excel
                    from io import BytesIO
                    excel_buffer = BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                        peer_matches.to_excel(writer, index=False, sheet_name='Peer Matches')
                    excel_data = excel_buffer.getvalue()
                    
                    st.download_button(
                        label="📊 Download Excel",
                        data=excel_data,
                        file_name='peer_tutoring_matches.xlsx',
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                    )
                
                with col_download3:
                    # Create PNG image of the table
                    import matplotlib.pyplot as plt
                    import matplotlib.patches as patches
                    from io import BytesIO
                    
                    fig, ax = plt.subplots(figsize=(12, len(peer_matches) * 0.5 + 2))
                    ax.axis('tight')
                    ax.axis('off')
                    
                    # Create table
                    table_data = peer_matches[['Subject', 'Tutor', 'Tutor_Score', 'Student', 'Student_Score', 'Score_Gap']].values
                    headers = ['Subject', 'Tutor', 'Tutor Score', 'Student', 'Student Score', 'Gap']
                    
                    table = ax.table(cellText=table_data, colLabels=headers, 
                                   cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
                    table.auto_set_font_size(False)
                    table.set_fontsize(9)
                    table.scale(1, 1.5)
                    
                    # Style the table
                    for i in range(len(headers)):
                        table[(0, i)].set_facecolor('#4CAF50')
                        table[(0, i)].set_text_props(weight='bold', color='white')
                    
                    plt.title('Peer Tutoring Matches', fontsize=16, fontweight='bold', pad=20)
                    
                    # Save to buffer
                    png_buffer = BytesIO()
                    plt.savefig(png_buffer, format='png', dpi=300, bbox_inches='tight')
                    png_data = png_buffer.getvalue()
                    plt.close()
                    
                    st.download_button(
                        label="🖼️ Download PNG",
                        data=png_data,
                        file_name='peer_tutoring_matches.png',
                        mime='image/png'
                    )
            else:
                # Enhanced explanations for no matches
                st.info("No peer matches generated.")
                
                # Analyze why no matches were created
                st.subheader("📋 Analysis: Why No Matches?")
                
                explanations = []
                for subject in ['physio', 'anatomy', 'bich']:
                    avg_col = f'{subject}_average'
                    if avg_col in df.columns:
                        subject_data = df[[df.columns[0], avg_col]].dropna()
                        
                        if len(subject_data) >= 2:
                            # Check struggling students
                            threshold_30_percent = int(len(subject_data) * 0.3)
                            struggling_by_rank = subject_data.tail(max(threshold_30_percent, 3))
                            struggling_by_score = subject_data[subject_data[avg_col] < 50]
                            struggling_names = set(struggling_by_rank.iloc[:, 0].tolist() + struggling_by_score.iloc[:, 0].tolist())
                            struggling_candidates = subject_data[subject_data.iloc[:, 0].isin(struggling_names)]
                            actual_struggling = struggling_candidates[struggling_candidates[avg_col] < 52]
                            
                            # Check potential tutors
                            potential_tutors = subject_data[subject_data[avg_col] > 75]
                            qualified_tutors_count = 0
                            
                            if len(actual_struggling) > 0:
                                avg_struggling_score = actual_struggling[avg_col].mean()
                                for _, tutor in potential_tutors.iterrows():
                                    if tutor[avg_col] > avg_struggling_score + 20:
                                        qualified_tutors_count += 1
                            
                            # Generate explanations
                            if len(actual_struggling) == 0:
                                explanations.append(f"**{subject.title()}**: No students scoring below 52% need tutoring")
                            elif len(potential_tutors) == 0:
                                min_score = subject_data[avg_col].min()
                                max_score = subject_data[avg_col].max()
                                explanations.append(f"**{subject.title()}**: No tutors available (highest score: {max_score:.1f}%, need >75%)")
                            elif qualified_tutors_count == 0:
                                avg_struggling = actual_struggling[avg_col].mean()
                                explanations.append(f"**{subject.title()}**: No qualified tutors (need 20+ point gap above {avg_struggling:.1f}%)")
                            else:
                                explanations.append(f"**{subject.title()}**: {len(actual_struggling)} students need help, {qualified_tutors_count} qualified tutors available")
                
                if explanations:
                    for explanation in explanations:
                        if "No students" in explanation:
                            st.success(explanation)
                        elif "No tutors" in explanation or "No qualified" in explanation:
                            st.warning(explanation)
                        else:
                            st.info(explanation)
                else:
                    st.warning("Upload data with multiple subjects to see detailed analysis.")
            
            # Detailed Subject Analysis
            st.header("📚 Subject-wise Analysis")
            
            available_subjects = []
            for subject in ['physio', 'anatomy', 'bich']:
                if f'{subject}_average' in df.columns:
                    available_subjects.append(subject)
            
            if available_subjects:
                col1, col2 = st.columns(2)
                
                with col1:
                    selected_subject = st.selectbox(
                        "Select subject for detailed analysis",
                        options=available_subjects
                    )
                
                with col2:
                    subject_sort_option = st.selectbox(
                        "Sort detailed scores by:",
                        options=['Average (High to Low)', 'Average (Low to High)', 'Name (A-Z)'],
                        index=0,
                        key='subject_sort'
                    )
                
                # Performance distribution for selected subject
                avg_col = f'{selected_subject}_average'
                scores = df[avg_col].dropna()
                
                if len(scores) > 0:
                    fig_hist = px.histogram(
                        x=scores,
                        title=f"{selected_subject.title()} Score Distribution",
                        labels={'x': 'Average Score (%)', 'y': 'Number of Students'}
                    )
                    st.plotly_chart(fig_hist, use_container_width=True,hide_index=True)
                    
                    # Show subject-specific data with sorting
                    st.subheader(f"📋 {selected_subject.title()} Detailed Scores")
                    subject_cols = [df.columns[0]]  # Name column
                    subject_cols.extend([col for col in df.columns 
                                       if selected_subject.lower() in str(col).lower()])
                    
                    subject_data = df[subject_cols].round(2).copy()
                    
                    # Apply sorting to subject data
                    if subject_sort_option == 'Average (High to Low)':
                        subject_data = subject_data.sort_values(avg_col, ascending=False)
                    elif subject_sort_option == 'Average (Low to High)':
                        subject_data = subject_data.sort_values(avg_col, ascending=True)
                    elif subject_sort_option == 'Name (A-Z)':
                        subject_data = subject_data.sort_values(df.columns[0], ascending=True)
                    
                    st.dataframe(subject_data, use_container_width=True,hide_index=True)
            else:
                st.info("No subject-specific analysis available. Ensure your columns contain subject names (Physio, Anatomy, Bich).")
            
            # Action Items
            st.header("🎯 Reading Group Action Items")
            
            high_risk_students = risk_df[risk_df['Risk_Level'].isin(['Critical Risk', 'High Risk'])]
            if not high_risk_students.empty:
                st.subheader("🚨 Immediate Attention Required:")
                for _, student in high_risk_students.iterrows():
                    declining_text = student['Declining_Subjects'] if student['Declining_Subjects'] else 'None'
                    risk_emoji = "💀" if student['Risk_Level'] == 'Critical Risk' else "🚨"
                    st.error(f"{risk_emoji} **{student['Student']}** ({student['Risk_Level']}) - Overall: {student['Overall_Average']}% | "
                           f"Declining in: {declining_text}")
            
            # Study focus recommendations
            if avg_cols:
                subject_priorities = []
                for subject in ['physio', 'anatomy', 'bich']:
                    avg_col = f'{subject}_average'
                    if avg_col in df.columns:
                        avg_score = df[avg_col].mean()
                        if not np.isnan(avg_score):
                            subject_priorities.append((subject, avg_score))
                
                if subject_priorities:
                    subject_priorities.sort(key=lambda x: x[1])  # Sort by score (lowest first)
                    
                    st.subheader("📖 Recommended Study Focus (Priority Order):")
                    for i, (subject, score) in enumerate(subject_priorities, 1):
                        if score < 50:
                            priority_emoji = "🔥"
                        elif score < 65:
                            priority_emoji = "⚠️"
                        else:
                            priority_emoji = "✅"
                        st.write(f"{i}. {priority_emoji} **{subject.title()}** (Group Average: {score:.1f}%)")
    
    else:
        st.info("👆 Please upload an Excel file to begin analysis")
        
        # Show sample format
        st.subheader("📋 Expected Excel Format:")
        st.write("Your Excel should have columns like these examples:")
        st.write("- **Name/Student columns:** Name, Student Name, Full Names, etc.")
        st.write("- **ID columns:** Matric, Matric No, Student ID, etc.")  
        st.write("- **Subject columns:** Physio CA1, Anatomy CA2, Bich Quiz 1, etc.")
        
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
        st.dataframe(pd.DataFrame(sample_data),hide_index=True)
        
        st.info("💡 **Tips:**\n- Column names are case-insensitive\n- Subjects can be abbreviated (Physio, Anatomy, Bich)\n- Assessment names can vary (CA1, CA2, Quiz 1, Test 2, etc.)")

if __name__ == "__main__":
    main()