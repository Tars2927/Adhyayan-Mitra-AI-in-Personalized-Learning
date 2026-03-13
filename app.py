import streamlit as st
import pandas as pd
import joblib
import google.generativeai as genai
import os
import plotly.graph_objects as go
import plotly.express as px

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Adhyayan Mitra · AI Learning Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;700&display=swap');

    /* ── Base ─────────────────────────────────── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: linear-gradient(145deg, #07071a 0%, #0f0f2e 50%, #130d2a 100%);
        min-height: 100vh;
    }

    /* ── Animated gradient keyframes ─────────── */
    @keyframes gradientShift {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    @keyframes pulse-ring {
        0%   { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(139,92,246,0.5); }
        70%  { transform: scale(1);    box-shadow: 0 0 0 14px rgba(139,92,246,0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(139,92,246,0); }
    }

    /* ── Header ──────────────────────────────── */
    .main-header {
        background: linear-gradient(270deg, #4f46e5, #7c3aed, #6366f1, #8b5cf6);
        background-size: 300% 300%;
        animation: gradientShift 8s ease infinite, fadeInUp 0.6s ease both;
        padding: 2.5rem 2.5rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 12px 40px rgba(99,102,241,0.4), inset 0 1px 0 rgba(255,255,255,0.15);
        position: relative;
        overflow: hidden;
    }

    .main-header::before {
        content: '';
        position: absolute;
        top: -50%; left: -50%;
        width: 200%; height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.05) 0%, transparent 60%);
        pointer-events: none;
    }

    .main-header h1 {
        color: #ffffff;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.6rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
        text-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }

    .main-header .tagline {
        color: rgba(255,255,255,0.88);
        font-size: 1.05rem;
        margin-top: 0.6rem;
        font-weight: 400;
        max-width: 620px;
    }

    .main-header .badges {
        margin-top: 1rem;
        display: flex;
        gap: 0.6rem;
        flex-wrap: wrap;
    }

    .badge {
        background: rgba(255,255,255,0.15);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255,255,255,0.2);
        color: #fff;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.3px;
    }

    /* ── Sidebar ─────────────────────────────── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #13132e 0%, #1e1e42 100%);
        border-right: 1px solid rgba(139,92,246,0.18);
    }

    [data-testid="stSidebar"] .sidebar-title {
        font-family: 'Space Grotesk', sans-serif;
        color: #c4b5fd;
        font-size: 1.25rem;
        font-weight: 700;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid rgba(99,102,241,0.5);
        margin-bottom: 1.2rem;
    }

    .sidebar-section {
        background: rgba(99,102,241,0.08);
        border: 1px solid rgba(99,102,241,0.18);
        border-radius: 12px;
        padding: 1rem 1rem 0.5rem;
        margin-bottom: 1rem;
    }

    .sidebar-section-title {
        color: #a5b4fc;
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.75rem;
    }

    .sidebar-tip {
        background: linear-gradient(135deg, rgba(59,130,246,0.12), rgba(99,102,241,0.1));
        border-left: 3px solid #6366f1;
        border-radius: 0 8px 8px 0;
        padding: 0.75rem 1rem;
        color: #c7d2fe;
        font-size: 0.85rem;
        line-height: 1.5;
        margin-top: 0.5rem;
    }

    /* ── Metric cards ────────────────────────── */
    .metric-card {
        background: linear-gradient(135deg, #1e1e42 0%, #252550 100%);
        padding: 1.4rem 1.6rem;
        border-radius: 16px;
        border-top: 3px solid transparent;
        margin-bottom: 0.75rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.35);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        position: relative;
        overflow: hidden;
        animation: fadeInUp 0.5s ease both;
    }

    .metric-card::after {
        content: '';
        position: absolute;
        inset: 0;
        background: linear-gradient(135deg, rgba(255,255,255,0.03) 0%, transparent 60%);
        pointer-events: none;
        border-radius: 16px;
    }

    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.4);
    }

    .metric-card.good  { border-top-color: #10b981; }
    .metric-card.warn  { border-top-color: #f59e0b; }
    .metric-card.bad   { border-top-color: #ef4444; }
    .metric-card.neutral { border-top-color: #8b5cf6; }

    .metric-icon {
        font-size: 1.6rem;
        margin-bottom: 0.4rem;
    }

    .metric-label {
        color: #94a3b8;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }

    .metric-value {
        color: #f1f5f9;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        margin-top: 0.3rem;
        line-height: 1;
    }

    .metric-sub {
        color: #64748b;
        font-size: 0.78rem;
        margin-top: 0.3rem;
    }

    /* ── Prediction result ───────────────────── */
    .prediction-box {
        background: linear-gradient(135deg, #4338ca 0%, #6d28d9 50%, #7c3aed 100%);
        padding: 2.5rem 2rem;
        border-radius: 20px;
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 0 12px 40px rgba(79,70,229,0.45);
        animation: fadeInUp 0.5s ease both;
        position: relative;
        overflow: hidden;
    }

    .prediction-box::before {
        content: '';
        position: absolute;
        top: -30%; right: -20%;
        width: 60%; height: 160%;
        background: radial-gradient(circle, rgba(255,255,255,0.08) 0%, transparent 65%);
        pointer-events: none;
    }

    .prediction-box h2 {
        color: rgba(255,255,255,0.85);
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .prediction-box .grade {
        color: #ffffff;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 4.5rem;
        font-weight: 800;
        text-shadow: 0 4px 16px rgba(0,0,0,0.35);
        line-height: 1;
    }

    .grade-letter {
        display: inline-block;
        background: rgba(255,255,255,0.18);
        border: 2px solid rgba(255,255,255,0.3);
        color: #fff;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.6rem;
        font-weight: 700;
        padding: 0.2rem 1rem;
        border-radius: 30px;
        margin-top: 0.75rem;
        animation: pulse-ring 2.5s ease-out infinite;
    }

    .grade-progress-wrap {
        background: rgba(255,255,255,0.12);
        border-radius: 30px;
        height: 10px;
        margin: 1.2rem auto 0;
        max-width: 280px;
        overflow: hidden;
    }

    .grade-progress-bar {
        height: 100%;
        border-radius: 30px;
        background: linear-gradient(90deg, #fbbf24, #34d399);
        transition: width 1s ease;
    }

    /* ── Comparison box ──────────────────────── */
    .comparison-box {
        background: linear-gradient(135deg, #065f46 0%, #059669 100%);
        padding: 1.8rem;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 6px 24px rgba(16,185,129,0.35);
    }

    .comparison-box.negative {
        background: linear-gradient(135deg, #7f1d1d 0%, #dc2626 100%);
        box-shadow: 0 6px 24px rgba(239,68,68,0.35);
    }

    .comparison-box.neutral-diff {
        background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%);
        box-shadow: 0 6px 24px rgba(37,99,235,0.35);
    }

    .comparison-box h3 {
        color: rgba(255,255,255,0.9);
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 0.6rem;
    }

    .comparison-box .grade {
        color: #fff;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 3rem;
        font-weight: 800;
    }

    .comparison-box .diff {
        color: rgba(255,255,255,0.85);
        font-size: 1.2rem;
        font-weight: 600;
        margin-top: 0.4rem;
    }

    /* ── Response container ──────────────────── */
    .response-container {
        background: rgba(30,30,66,0.7);
        backdrop-filter: blur(12px);
        padding: 2rem 2.2rem;
        border-radius: 18px;
        border: 1px solid rgba(139,92,246,0.25);
        margin-top: 1.5rem;
        box-shadow: 0 4px 24px rgba(0,0,0,0.25);
        animation: fadeInUp 0.5s ease both;
    }

    .response-container h3 {
        color: #c4b5fd;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 0.75rem;
    }

    /* ── Tabs ────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: rgba(30,30,66,0.5);
        border-radius: 14px;
        padding: 0.5rem;
        border: 1px solid rgba(99,102,241,0.18);
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 10px;
        color: #94a3b8;
        font-weight: 600;
        font-size: 0.9rem;
        padding: 0.65rem 1.3rem;
        transition: all 0.2s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: #c4b5fd;
        background: rgba(99,102,241,0.1);
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(120deg, #6366f1 0%, #8b5cf6 100%) !important;
        color: white !important;
        box-shadow: 0 4px 12px rgba(99,102,241,0.4);
    }

    /* ── Buttons ─────────────────────────────── */
    .stButton > button {
        background: linear-gradient(120deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        font-size: 1rem;
        padding: 0.75rem 2rem;
        border-radius: 12px;
        border: none;
        box-shadow: 0 4px 16px rgba(99,102,241,0.45);
        transition: all 0.25s ease;
        width: 100%;
        letter-spacing: 0.3px;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(99,102,241,0.6);
        background: linear-gradient(120deg, #4f46e5 0%, #7c3aed 100%);
    }

    .stButton > button:active {
        transform: translateY(0);
    }

    /* ── Slider ──────────────────────────────── */
    .stSlider [data-baseweb="slider"] [role="slider"] {
        background: #8b5cf6;
        border: 3px solid #c4b5fd;
        width: 20px; height: 20px;
    }

    /* ── Info / tip boxes ────────────────────── */
    .info-box {
        background: linear-gradient(135deg, rgba(59,130,246,0.1), rgba(99,102,241,0.08));
        border-left: 3px solid #6366f1;
        padding: 0.85rem 1rem;
        border-radius: 0 10px 10px 0;
        margin: 0.75rem 0;
        color: #c7d2fe;
        font-size: 0.88rem;
        line-height: 1.5;
    }

    /* ── Quiz container ──────────────────────── */
    .quiz-container {
        background: rgba(30,30,66,0.7);
        backdrop-filter: blur(12px);
        padding: 2rem 2.2rem;
        border-radius: 18px;
        border: 1px solid rgba(139,92,246,0.25);
        margin-top: 1rem;
        box-shadow: 0 4px 24px rgba(0,0,0,0.25);
    }

    /* ── What-if section ─────────────────────── */
    .whatif-section {
        background: rgba(30,30,66,0.45);
        padding: 1.5rem 1.8rem;
        border-radius: 16px;
        border: 2px dashed rgba(139,92,246,0.35);
        margin: 1rem 0;
    }

    /* ── Grade scale reference ───────────────── */
    .grade-scale {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
        margin-top: 0.75rem;
    }

    .grade-chip {
        padding: 0.3rem 0.75rem;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.3px;
    }

    .chip-a  { background: rgba(16,185,129,0.2); color: #34d399; border: 1px solid rgba(16,185,129,0.3); }
    .chip-b  { background: rgba(99,102,241,0.2); color: #a5b4fc; border: 1px solid rgba(99,102,241,0.3); }
    .chip-c  { background: rgba(245,158,11,0.2); color: #fcd34d; border: 1px solid rgba(245,158,11,0.3); }
    .chip-d  { background: rgba(239,68,68,0.2);  color: #fca5a5; border: 1px solid rgba(239,68,68,0.3); }

    /* ── Section headings ────────────────────── */
    .section-heading {
        color: #e0e7ff;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
    }

    .section-sub {
        color: #64748b;
        font-size: 0.9rem;
        margin-bottom: 1.5rem;
    }

    /* ── Insight row ─────────────────────────── */
    .insight-row {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.65rem 0;
        border-bottom: 1px solid rgba(99,102,241,0.12);
    }

    .insight-row:last-child { border-bottom: none; }

    .insight-label {
        color: #94a3b8;
        font-size: 0.88rem;
        flex: 1;
    }

    .insight-bar-wrap {
        flex: 2;
        background: rgba(255,255,255,0.06);
        border-radius: 20px;
        height: 8px;
        overflow: hidden;
    }

    .insight-bar {
        height: 100%;
        border-radius: 20px;
        transition: width 0.8s ease;
    }

    .insight-score {
        color: #e0e7ff;
        font-weight: 700;
        font-size: 0.88rem;
        min-width: 42px;
        text-align: right;
    }

    /* ── Footer ──────────────────────────────── */
    .footer {
        text-align: center;
        padding: 1.5rem;
        margin-top: 1rem;
        border-top: 1px solid rgba(99,102,241,0.18);
        color: #475569;
        font-size: 0.85rem;
    }

    .footer a { color: #818cf8; text-decoration: none; }
    .footer a:hover { text-decoration: underline; }

    /* ── Streamlit overrides ─────────────────── */
    .stSelectbox label, .stSlider label, .stTextInput label {
        color: #94a3b8 !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(99,102,241,0.2);
    }

    .stSpinner > div {
        border-top-color: #8b5cf6 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- API & MODEL CONFIGURATION ---
try:
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

    if not GEMINI_API_KEY:
        st.error("⚠️ GEMINI_API_KEY is not set. Please configure your API key.")
        st.stop()

    genai.configure(api_key=GEMINI_API_KEY)
    llm = genai.GenerativeModel('gemini-2.5-pro')
    model = joblib.load('student_model.pkl')

except Exception as e:
    st.error(f"❌ An error occurred during initialization: {e}")
    st.stop()

# --- CONSTANTS ---
MODEL_COLUMNS = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures',
                 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences',
                 'G1', 'G2', 'school_MS', 'sex_M', 'address_U', 'famsize_LE3',
                 'Pstatus_T', 'Mjob_health', 'Mjob_other', 'Mjob_services',
                 'Mjob_teacher', 'Fjob_health', 'Fjob_other', 'Fjob_services',
                 'Fjob_teacher', 'reason_home', 'reason_other', 'reason_reputation',
                 'guardian_mother', 'guardian_other', 'schoolsup_yes', 'famsup_yes',
                 'paid_yes', 'activities_yes', 'nursery_yes', 'higher_yes',
                 'internet_yes', 'romantic_yes']

# --- HELPER FUNCTIONS ---
def grade_letter(grade_20):
    """Return a tuple of (label, hex colour, CSS chip class) for a score out of 20 (Portuguese scale).
    
    Labels: 'A (Excellent)', 'B (Good)', 'C (Satisfactory)', 'D (Needs Improvement)'
    """
    if grade_20 >= 17:
        return "A (Excellent)", "#10b981", "chip-a"
    elif grade_20 >= 14:
        return "B (Good)", "#818cf8", "chip-b"
    elif grade_20 >= 10:
        return "C (Satisfactory)", "#f59e0b", "chip-c"
    else:
        return "D (Needs Improvement)", "#ef4444", "chip-d"

def metric_color(factor, value):
    """Return a CSS class name ('good', 'warn', 'bad', or 'neutral') for a metric card.

    Parameters
    ----------
    factor : str
        One of 'studytime', 'failures', 'goout', or 'absences'.
    value : int | float
        The current value of that factor.
    """
    if factor == 'studytime':
        if value >= 3: return 'good'
        if value == 2: return 'warn'
        return 'bad'
    if factor == 'failures':
        if value == 0: return 'good'
        if value <= 1: return 'warn'
        return 'bad'
    if factor == 'goout':
        if value <= 3: return 'good'
        if value == 4: return 'warn'
        return 'bad'
    if factor == 'absences':
        if value <= 5: return 'good'
        if value <= 15: return 'warn'
        return 'bad'
    return 'neutral'

def bar_color(score):
    """Return a hex colour for an insight bar based on score (0–100)."""
    if score >= 70:
        return '#10b981'
    if score >= 45:
        return '#f59e0b'
    return '#ef4444'

def predict_grade(studytime, failures, goout, absences):
    """Make a grade prediction based on input parameters"""
    input_df = pd.DataFrame(0, index=[0], columns=MODEL_COLUMNS)
    input_df['studytime'] = studytime
    input_df['failures'] = failures
    input_df['goout'] = goout
    input_df['absences'] = absences
    input_df['age'] = 17
    input_df['G1'] = 11
    input_df['G2'] = 11
    return model.predict(input_df)[0]

def create_sensitivity_chart(base_studytime, base_failures, base_goout, base_absences):
    """Create a bar chart showing impact of changing each variable"""
    base_grade = predict_grade(base_studytime, base_failures, base_goout, base_absences)
    
    # Calculate impact of improving each factor
    impacts = []
    
    # Study time impact (increase by 1)
    if base_studytime < 4:
        new_grade = predict_grade(base_studytime + 1, base_failures, base_goout, base_absences)
        impacts.append({
            'Factor': 'Study Time +1',
            'Impact': new_grade - base_grade,
            'Description': 'Increase weekly study time'
        })
    
    # Reduce failures (if any)
    if base_failures > 0:
        new_grade = predict_grade(base_studytime, base_failures - 1, base_goout, base_absences)
        impacts.append({
            'Factor': 'Fewer Failures',
            'Impact': new_grade - base_grade,
            'Description': 'Reduce past failures by 1'
        })
    
    # Reduce socializing (if high)
    if base_goout > 2:
        new_grade = predict_grade(base_studytime, base_failures, base_goout - 1, base_absences)
        impacts.append({
            'Factor': 'Less Socializing',
            'Impact': new_grade - base_grade,
            'Description': 'Reduce going out by 1 level'
        })
    
    # Reduce absences
    if base_absences > 5:
        new_grade = predict_grade(base_studytime, base_failures, base_goout, max(0, base_absences - 5))
        impacts.append({
            'Factor': 'Better Attendance',
            'Impact': new_grade - base_grade,
            'Description': 'Reduce absences by 5'
        })
    
    df = pd.DataFrame(impacts)
    
    if len(df) > 0:
        df = df.sort_values('Impact', ascending=True)
        
        fig = go.Figure()
        
        colors = ['#10b981' if x > 0 else '#ef4444' for x in df['Impact']]
        
        fig.add_trace(go.Bar(
            y=df['Factor'],
            x=df['Impact'],
            orientation='h',
            marker=dict(color=colors, line=dict(color='rgba(255,255,255,0.3)', width=1)),
            text=[f"+{x:.2f}" if x > 0 else f"{x:.2f}" for x in df['Impact']],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Grade Impact: %{x:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(
                text='Impact of Habit Changes on Your Grade',
                font=dict(size=18, color='#ffffff', family='Arial, sans-serif'),
                x=0.5,
                xanchor='center'
            ),
            xaxis=dict(
                title='Grade Point Change',
                gridcolor='rgba(165, 180, 252, 0.2)',
                zerolinecolor='rgba(165, 180, 252, 0.4)',
                tickfont=dict(color='#a5b4fc')
            ),
            yaxis=dict(
                tickfont=dict(color='#ffffff', size=12)
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(30, 30, 63, 0.3)',
            font=dict(color='#ffffff'),
            height=300,
            margin=dict(l=20, r=80, t=60, b=40),
            showlegend=False
        )
        
        return fig, df
    else:
        return None, None

def create_radar_chart(studytime, failures, goout, absences):
    """Create a radar chart to visualize student habits"""
    study_score = (studytime / 4) * 100
    failure_score = 100 - ((failures / 4) * 100)
    social_score = (goout / 5) * 100
    attendance_score = max(0, 100 - (absences / 93) * 100)
    
    categories = ['Study Time', 'Academic Success', 'Social Balance', 'Attendance']
    values = [study_score, failure_score, social_score, attendance_score]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(139, 92, 246, 0.3)',
        line=dict(color='rgb(139, 92, 246)', width=3),
        marker=dict(size=8, color='rgb(139, 92, 246)'),
        name='Your Profile'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                showticklabels=True,
                ticks='',
                gridcolor='rgba(165, 180, 252, 0.3)',
                tickfont=dict(color='#a5b4fc', size=10)
            ),
            angularaxis=dict(
                gridcolor='rgba(165, 180, 252, 0.3)',
                linecolor='rgba(165, 180, 252, 0.3)',
                tickfont=dict(color='#ffffff', size=12, family='Arial, sans-serif')
            ),
            bgcolor='rgba(30, 30, 63, 0.5)'
        ),
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#ffffff'),
        height=400,
        margin=dict(l=80, r=80, t=40, b=40)
    )
    
    return fig

def create_master_prompt(student_data, predicted_grade, style):
    prompt = f"""
    You are an expert AI academic advisor. Your tone is supportive and encouraging.

    A student has provided the following information:
    - Weekly Study Time (1-4): {student_data['studytime']}
    - Past Class Failures: {student_data['failures']}
    - Socializing (1-5): {student_data['goout']}
    - Absences: {student_data['absences']}
    - Preferred Learning Style: {style}

    Our predictive model estimates their final grade will be {predicted_grade:.2f} out of 20.

    Your task is to provide two things in your response:

    1.  **Personalized Advice:** Write a short paragraph of feedback (3-4 sentences) tailored to the student's **{style}** learning style.
    2.  **Actionable Resource:** Recommend one specific, real, and free online resource (like a YouTube video, a website, or a free app) that aligns with their **{style}**.

    Structure your response with markdown headings for "Personalized Advice" and "Recommended Resource".
    """
    return prompt

def generate_quiz(topic, difficulty, num_questions):
    """Generate a custom quiz based on user input"""
    prompt = f"""
    Create a quiz on the topic: **{topic}**
    
    Requirements:
    - Difficulty level: {difficulty}
    - Number of questions: {num_questions}
    - Format: Multiple choice with 4 options (A, B, C, D)
    - Include the correct answer at the end of each question
    - Make questions engaging and educational
    
    Structure each question as follows:
    **Question X:**
    [Question text]
    
    A) [Option A]
    B) [Option B]
    C) [Option C]
    D) [Option D]
    
    **Correct Answer:** [Letter]
    **Explanation:** [Brief explanation why this is correct]
    
    ---
    
    Start generating the quiz now!
    """
    return prompt

# --- HEADER ---
st.markdown("""
<div class="main-header">
    <h1>🎓 Adhyayan Mitra</h1>
    <div class="tagline">Your AI-powered academic companion — predict performance, explore potential, and receive personalised learning guidance.</div>
    <div class="badges">
        <span class="badge">⚡ Powered by Gemini AI</span>
        <span class="badge">📊 ML Grade Predictor</span>
        <span class="badge">🎯 Learning Style Adaptive</span>
        <span class="badge">🔮 What-If Simulator</span>
    </div>
</div>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown('<div class="sidebar-title">📊 Student Profile</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-title">📚 Academic Habits</div>', unsafe_allow_html=True)
    studytime = st.slider('Weekly Study Time', 1, 4, 2,
                          help="1 = <2 hrs/wk · 2 = 2–5 hrs · 3 = 5–10 hrs · 4 = >10 hrs")
    failures = st.slider('Past Class Failures', 0, 4, 0,
                        help="Number of previous academic failures (0 is ideal)")
    absences = st.slider('School Absences', 0, 93, 5,
                        help="Total absences this term (lower is better)")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-title">👥 Social Life</div>', unsafe_allow_html=True)
    goout = st.slider('Socialising Frequency', 1, 5, 3,
                     help="1 = Very low · 5 = Very high")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-title">🧠 Learning Style</div>', unsafe_allow_html=True)
    learning_style = st.selectbox(
        "Preferred Learning Style",
        ("Visual", "Auditory", "Reading/Writing", "Kinesthetic"),
        help="Choose the style that works best for you"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-tip">
        💡 <strong>Tip:</strong> Be honest with your inputs — accurate data leads to more useful personalised advice!
    </div>
    """, unsafe_allow_html=True)

# --- MAIN CONTENT WITH TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["📈 Grade Prediction", "🔮 What-If Analysis", "🎯 Profile Visualization", "📝 Quiz Generator"])

# TAB 1: GRADE PREDICTION
with tab1:
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        c = metric_color('studytime', studytime)
        study_labels = {1: '<2 hrs/wk', 2: '2–5 hrs/wk', 3: '5–10 hrs/wk', 4: '>10 hrs/wk'}
        st.markdown(f"""
        <div class="metric-card {c}">
            <div class="metric-icon">📚</div>
            <div class="metric-label">Study Time</div>
            <div class="metric-value">{studytime}/4</div>
            <div class="metric-sub">{study_labels[studytime]}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        c = metric_color('failures', failures)
        st.markdown(f"""
        <div class="metric-card {c}">
            <div class="metric-icon">{'✅' if failures == 0 else '⚠️'}</div>
            <div class="metric-label">Past Failures</div>
            <div class="metric-value">{failures}</div>
            <div class="metric-sub">{'None — great!' if failures == 0 else f'{failures} failure{"s" if failures>1 else ""}'}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        c = metric_color('goout', goout)
        social_labels = {1: 'Very low', 2: 'Low', 3: 'Moderate', 4: 'High', 5: 'Very high'}
        st.markdown(f"""
        <div class="metric-card {c}">
            <div class="metric-icon">👥</div>
            <div class="metric-label">Social Level</div>
            <div class="metric-value">{goout}/5</div>
            <div class="metric-sub">{social_labels[goout]}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        c = metric_color('absences', absences)
        st.markdown(f"""
        <div class="metric-card {c}">
            <div class="metric-icon">🗓️</div>
            <div class="metric-label">Absences</div>
            <div class="metric-value">{absences}</div>
            <div class="metric-sub">{'Great attendance!' if absences <= 5 else ('Some absences' if absences <= 15 else 'High — review needed')}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🚀 Generate My Personalised Plan", key="predict_btn"):
        predicted_grade = predict_grade(studytime, failures, goout, absences)
        letter, color, chip_cls = grade_letter(predicted_grade)
        pct = min(100, max(0, (predicted_grade / 20) * 100))

        st.markdown(f"""
        <div class="prediction-box">
            <h2>🎯 Predicted Final Grade</h2>
            <div class="grade">{predicted_grade:.1f} <span style="font-size:1.6rem;opacity:0.7">/ 20</span></div>
            <div class="grade-letter" style="background:rgba(255,255,255,0.18); border-color:{color}; color:#fff;">{letter}</div>
            <div class="grade-progress-wrap">
                <div class="grade-progress-bar" style="width:{pct:.0f}%;"></div>
            </div>
            <div style="color:rgba(255,255,255,0.6); font-size:0.8rem; margin-top:0.6rem;
                        display:flex; justify-content:space-between; max-width:280px; margin-left:auto; margin-right:auto;">
                <span>0</span><span>20</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        input_df = pd.DataFrame(0, index=[0], columns=MODEL_COLUMNS)
        input_df['studytime'] = studytime
        input_df['failures'] = failures
        input_df['goout'] = goout
        input_df['absences'] = absences

        with st.spinner("✨ Crafting your personalised learning plan..."):
            master_prompt = create_master_prompt(input_df.iloc[0], predicted_grade, learning_style)
            response = llm.generate_content(master_prompt)

            st.markdown('<div class="response-container">', unsafe_allow_html=True)
            st.markdown("## 📝 Your Personalised Learning Plan")
            st.markdown(response.text)
            st.markdown('</div>', unsafe_allow_html=True)

    # Grade scale reference
    st.markdown("""
    <br>
    <div style="color:#475569; font-size:0.8rem; font-weight:600; text-transform:uppercase; letter-spacing:0.8px; margin-bottom:0.5rem;">
        Grade Scale Reference (Portuguese system, out of 20)
    </div>
    <div class="grade-scale">
        <span class="grade-chip chip-a">A · 17–20 · Excellent</span>
        <span class="grade-chip chip-b">B · 14–16 · Good</span>
        <span class="grade-chip chip-c">C · 10–13 · Satisfactory</span>
        <span class="grade-chip chip-d">D · 0–9 · Needs Improvement</span>
    </div>
    """, unsafe_allow_html=True)

# TAB 2: WHAT-IF ANALYSIS
with tab2:
    st.markdown('<div class="section-heading">🔮 What-If Analysis: Explore Your Potential</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">See how changing your habits could shift your predicted grade. Experiment freely!</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Current prediction
    current_grade = predict_grade(studytime, failures, goout, absences)
    curr_letter, curr_color, _ = grade_letter(current_grade)
    curr_pct = min(100, max(0, (current_grade / 20) * 100))

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown(f"""
        <div class="prediction-box">
            <h2>📊 Current Prediction</h2>
            <div class="grade">{current_grade:.1f} <span style="font-size:1.4rem;opacity:0.7">/ 20</span></div>
            <div class="grade-letter" style="border-color:{curr_color};">{curr_letter}</div>
            <div class="grade-progress-wrap">
                <div class="grade-progress-bar" style="width:{curr_pct:.0f}%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Sensitivity analysis
    st.markdown("### 📈 Impact of Improving Your Habits")
    sensitivity_fig, sensitivity_df = create_sensitivity_chart(studytime, failures, goout, absences)
    
    if sensitivity_fig:
        st.plotly_chart(sensitivity_fig, use_container_width=True)
        
        st.markdown('<div class="response-container">', unsafe_allow_html=True)
        st.markdown("#### 💡 Key Insights")
        
        if len(sensitivity_df) > 0:
            best_improvement = sensitivity_df.iloc[-1]
            st.markdown(f"""
            <p style='color: #e0e7ff; font-size: 1.1rem;'>
            <strong>Best opportunity for improvement:</strong> {best_improvement['Description']} 
            could increase your grade by <strong style='color: #10b981;'>+{best_improvement['Impact']:.2f} points</strong>!
            </p>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("🎉 You're already at optimal levels! Keep up the great work!")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Interactive What-If Simulator
    st.markdown("### 🎮 Interactive Scenario Simulator")
    st.markdown('<div class="whatif-section">', unsafe_allow_html=True)
    st.markdown("**Adjust the sliders below to test different scenarios:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        whatif_studytime = st.slider('What if my study time was...', 1, 4, studytime, key='whatif_study')
        whatif_failures = st.slider('What if I had... failures', 0, 4, failures, key='whatif_fail')
    
    with col2:
        whatif_goout = st.slider('What if I went out... often', 1, 5, goout, key='whatif_goout')
        whatif_absences = st.slider('What if I had... absences', 0, 93, absences, key='whatif_abs')
    
    whatif_grade = predict_grade(whatif_studytime, whatif_failures, whatif_goout, whatif_absences)
    grade_diff = whatif_grade - current_grade
    whatif_letter, whatif_color, _ = grade_letter(whatif_grade)
    whatif_pct = min(100, max(0, (whatif_grade / 20) * 100))

    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 1])

    with col2:
        if grade_diff > 0:
            diff_class = ""
        elif grade_diff < 0:
            diff_class = "negative"
        else:
            diff_class = "neutral-diff"
        sign = "+" if grade_diff >= 0 else ""
        emoji = "📈" if grade_diff > 0 else "📉" if grade_diff < 0 else "➡️"

        st.markdown(f"""
        <div class="comparison-box {diff_class}">
            <h3>{emoji} Scenario Result</h3>
            <div class="grade">{whatif_grade:.1f} / 20</div>
            <div style="color:rgba(255,255,255,0.75); font-size:0.85rem; margin:0.3rem 0;">{whatif_letter}</div>
            <div class="diff">{sign}{grade_diff:.2f} pts</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Comparison table
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📊 Detailed Comparison")
    
    comparison_data = {
        'Metric': ['Study Time', 'Past Failures', 'Going Out', 'Absences', '**Predicted Grade**'],
        'Current': [studytime, failures, goout, absences, f"**{current_grade:.2f}**"],
        'Scenario': [whatif_studytime, whatif_failures, whatif_goout, whatif_absences, f"**{whatif_grade:.2f}**"],
        'Change': [
            f"{whatif_studytime - studytime:+d}",
            f"{whatif_failures - failures:+d}",
            f"{whatif_goout - goout:+d}",
            f"{whatif_absences - absences:+d}",
            f"**{grade_diff:+.2f}**"
        ]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

# TAB 3: STUDENT PROFILE VISUALIZATION
with tab3:
    st.markdown('<div class="section-heading">🎯 Your Learning Profile</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">A visualisation of your strengths and areas for improvement across key academic dimensions.</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    radar_fig = create_radar_chart(studytime, failures, goout, absences)
    st.plotly_chart(radar_fig, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Compute dimension scores for the insight bars
    study_score     = int((studytime / 4) * 100)
    success_score   = int(100 - (failures / 4) * 100)
    social_score    = int((goout / 5) * 100)
    attendance_score = int(max(0, 100 - (absences / 93) * 100))

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div class="response-container">
            <h3>📊 Dimension Breakdown</h3>
            <div class="insight-row">
                <span class="insight-label">📚 Study Time</span>
                <div class="insight-bar-wrap">
                    <div class="insight-bar" style="width:{study_score}%; background:{bar_color(study_score)};"></div>
                </div>
                <span class="insight-score">{study_score}%</span>
            </div>
            <div class="insight-row">
                <span class="insight-label">🏆 Academic Success</span>
                <div class="insight-bar-wrap">
                    <div class="insight-bar" style="width:{success_score}%; background:{bar_color(success_score)};"></div>
                </div>
                <span class="insight-score">{success_score}%</span>
            </div>
            <div class="insight-row">
                <span class="insight-label">👥 Social Balance</span>
                <div class="insight-bar-wrap">
                    <div class="insight-bar" style="width:{social_score}%; background:{bar_color(social_score)};"></div>
                </div>
                <span class="insight-score">{social_score}%</span>
            </div>
            <div class="insight-row">
                <span class="insight-label">🗓️ Attendance</span>
                <div class="insight-bar-wrap">
                    <div class="insight-bar" style="width:{attendance_score}%; background:{bar_color(attendance_score)};"></div>
                </div>
                <span class="insight-score">{attendance_score}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        overall = int((study_score + success_score + attendance_score) / 3)
        insights = []
        if study_score < 50:
            insights.append("⬆️ Increasing study time is your biggest lever for improvement.")
        if success_score < 75:
            insights.append("📖 Addressing past failures through extra practice can significantly boost your grade.")
        if social_score > 80:
            insights.append("⚖️ Your social activity is very high — a small reduction could free up study time.")
        if attendance_score < 70:
            insights.append("🗓️ Improving your attendance has a direct impact on learning outcomes.")
        if not insights:
            insights.append("🎉 Your profile looks well-balanced — keep up the excellent habits!")

        st.markdown(f"""
        <div class="response-container">
            <h3>💡 Personalised Insights</h3>
            <div style="color:#94a3b8; font-size:0.78rem; text-transform:uppercase; letter-spacing:0.8px; margin-bottom:0.6rem;">
                Overall Academic Score
            </div>
            <div style="font-family:'Space Grotesk',sans-serif; font-size:2.8rem; font-weight:800; color:#e0e7ff; margin-bottom:0.8rem; line-height:1;">
                {overall}<span style="font-size:1.2rem; color:#64748b;"> / 100</span>
            </div>
            {''.join(f'<p style="color:#c7d2fe; font-size:0.9rem; line-height:1.6; margin:0.5rem 0;">{i}</p>' for i in insights)}
        </div>
        """, unsafe_allow_html=True)

# TAB 4: QUIZ GENERATOR
with tab4:
    st.markdown('<div class="section-heading">📝 Dynamic Quiz Generator</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Generate custom quizzes on any topic to test and reinforce your knowledge.</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        quiz_topic = st.text_input(
            "🎯 Enter a topic",
            placeholder="e.g., Python Programming, World History, Biology...",
            help="Type any subject you want to be quizzed on"
        )
    
    with col2:
        num_questions = st.selectbox(
            "Number of Questions",
            [3, 5, 10, 15],
            index=1
        )
    
    difficulty = st.select_slider(
        "Difficulty Level",
        options=["Beginner", "Intermediate", "Advanced"],
        value="Intermediate"
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🎲 Generate Quiz", key="quiz_btn"):
        if quiz_topic.strip():
            with st.spinner(f"✨ Generating your {difficulty.lower()} quiz on {quiz_topic}..."):
                quiz_prompt = generate_quiz(quiz_topic, difficulty, num_questions)
                quiz_response = llm.generate_content(quiz_prompt)
                
                st.markdown('<div class="quiz-container">', unsafe_allow_html=True)
                st.markdown(f"### 📚 Quiz: {quiz_topic}")
                st.markdown(f"**Difficulty:** {difficulty} | **Questions:** {num_questions}")
                st.markdown("---")
                st.markdown(quiz_response.text)
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("⚠️ Please enter a topic for your quiz!")

# --- FOOTER ---
st.markdown("""
<div class="footer">
    Made with ❤️ by <strong style="color:#818cf8;">TARS</strong> &nbsp;·&nbsp;
    Powered by <strong style="color:#818cf8;">Google Gemini</strong> &amp; <strong style="color:#818cf8;">Streamlit</strong> &nbsp;·&nbsp;
    Built for Student Success 🎓
    <br>
    <span style="font-size:0.78rem; color:#334155; margin-top:0.3rem; display:block;">
        Dataset: UCI Student Performance (Cortez &amp; Silva, 2008)
    </span>
</div>
""", unsafe_allow_html=True)