import streamlit as st

def apply_custom_styling():
    """Applies the premium PetroAI Suite styling (CSS) to the Streamlit app."""
    st.markdown("""
    <style>
        /* Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Outfit', sans-serif;
        }

        /* Gradient Background */
        .stApp {
            background: radial-gradient(circle at 50% 50%, #1e293b 0%, #0f172a 100%);
            color: #f8fafc;
        }

        /* Main Container Padding */
        .main > div {
            padding-top: 1rem;
        }

        /* GLASSMORPHISM CARDS */
        .stTabs [data-baseweb="tab-list"] {
            background: rgba(30, 41, 59, 0.4);
            padding: 0.5rem;
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.05);
            gap: 10px;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            border-radius: 8px;
            transition: all 0.2s ease;
            padding: 0 20px;
            color: #94a3b8;
        }

        .stTabs [aria-selected="true"] {
            background: rgba(56, 189, 248, 0.1) !important;
            color: #38bdf8 !important;
            border: 1px solid rgba(56, 189, 248, 0.2) !important;
        }

        /* METRICS NEON GLOW */
        div[data-testid="stMetric"] {
            background: rgba(30, 41, 59, 0.6);
            backdrop-filter: blur(16px) saturate(180%);
            -webkit-backdrop-filter: blur(16px) saturate(180%);
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 1.8rem;
            border-radius: 20px;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5);
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        }
        
        div[data-testid="stMetric"]:hover {
            transform: scale(1.05);
            border-color: rgba(56, 189, 248, 0.5);
            box-shadow: 0 0 25px rgba(56, 189, 248, 0.15);
        }

        /* BUTTON NEON */
        .stButton>button {
            background: linear-gradient(135deg, #38bdf8 0%, #1d4ed8 100%);
            color: white;
            border: none;
            padding: 0.8rem 2.5rem;
            border-radius: 14px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
            box-shadow: 0 0 10px rgba(56, 189, 248, 0.3);
            transition: all 0.3s ease;
        }
        
        .stButton>button:hover {
            box-shadow: 0 0 30px rgba(56, 189, 248, 0.7);
            transform: translateY(-2px);
        }

        /* SIDEBAR GLASS */
        section[data-testid="stSidebar"] {
            background: rgba(15, 23, 42, 0.95);
            border-right: 1px solid rgba(255, 255, 255, 0.05);
        }

        /* TITLES WITH NEON TEXT */
        h1 {
            font-size: 3.5rem !important;
            background: linear-gradient(45deg, #38bdf8, #818cf8, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 900 !important;
            text-align: center;
            margin-bottom: 0.5rem !important;
            animation: fadeIn 1.2s ease-out;
        }
        
        h2, h3 {
            color: #f1f5f9;
            font-weight: 700 !important;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            padding-bottom: 0.5rem;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* FILE UPLOADER CUSTOM */
        [data-testid="stFileUploader"] {
            background: rgba(30, 41, 59, 0.6);
            border-radius: 16px;
            padding: 10px;
            border: 2px dashed rgba(56, 189, 248, 0.3);
        }

        /* Status Bar Custom */
        div[data-testid="stStatusWidget"] {
            background: #0f172a;
            color: #38bdf8;
        }
    </style>
    """, unsafe_allow_html=True)

def get_plotly_layout(title, x_title="", y_title=""):
    """Returns a premium Dark Matter themed Plotly layout."""
    return dict(
        title=dict(text=f"<b>{title}</b>", font=dict(size=20, color="#f8fafc")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(30, 41, 59, 0.2)",
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.05)",
            title=dict(text=x_title, font=dict(color="#94a3b8")),
            tickfont=dict(color="#94a3b8"),
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.05)",
            title=dict(text=y_title, font=dict(color="#94a3b8")),
            tickfont=dict(color="#94a3b8"),
        ),
        font=dict(family="Outfit, sans-serif", color="#f8fafc"),
        hovermode="x unified",
        margin=dict(t=50, b=50, l=50, r=50)
    )
