# Member 1: UI & Frontend

This directory contains the Streamlit application interface for **LawBot** - an AI RAG Assistant tailored for Vietnamese Drug Laws and News.

## 🚀 Ultra Premium UI (V5) Features

The UI has been heavily customized to push Streamlit beyond its default capabilities, providing a seamless "Dark Academic / AI Dashboard" experience.

1. **Glassmorphism Design:** Cards, sidebars, and expanders use translucent backgrounds with `backdrop-filter: blur()`, giving the app a futuristic, multi-layered depth.
2. **Animated Mesh Background:** A subtle, slow-moving radial gradient background that shifts colors to make the app feel alive.
3. **Advanced Micro-animations:** Premium glowing typing indicators, hover-lift effects on source cards, and animated glowing buttons.
4. **Answer Quality Indicators:** Every RAG response includes a metadata strip showing generation metrics (e.g., *LLM generation*, *Source count*, *Retrieval method*).
5. **Structured Source Cards:** Cited documents are rendered as beautiful, distinct cards with citation badges (e.g., `[S1]`, `[S2]`) and score indicators, rather than plain text.
6. **Component Architecture:** The code in `app.py` is fully modularized into functions (`render_header()`, `render_sidebar()`, etc.) for easy maintenance.

## 📁 Structure

- `app.py`: The main Streamlit entrypoint and UI rendering logic.
- `styles.css`: The massive custom CSS file defining the Ultra Premium Design Tokens, Glassmorphism effects, and Animations.
- `README.md`: This file.

## ⚙️ How to Run

Because you might be using a virtual environment, running `streamlit run` directly might result in a "Command not found" error if the environment isn't activated.

**Best way to run:**
From the `group_project` root folder, run:

```bash
python -m streamlit run app.py
```

This bypasses PATH issues and directly invokes Streamlit using your active Python interpreter.

## 🎨 Modifying the Theme
All styling is centralized in `styles.css` under the `:root` pseudo-class. 
- Primary Color: `--primary: #7c3aed;` (Purple)
- Secondary Color: `--secondary: #3b82f6;` (Blue)
- Background Base: `--bg-deep: #050511;`

You can tweak these hex codes in `styles.css` to instantly change the entire application's color scheme!
