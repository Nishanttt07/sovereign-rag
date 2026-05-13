title Team Roles and Responsibilities
direction right

// Nodes and groups
Project Start [shape: oval, color: lightblue, icon: flag]

Backend Development [color: blue, icon: server] {
  Search Algorithms [icon: search, color: blue]
  Query Routing [icon: git-branch, color: blue]
  AI Integrations [icon: cpu, color: blue]
}

UI Development [color: green, icon: layout] {
  Streamlit Interface [icon: monitor, color: green]
  File Ingestion [icon: upload, color: green]
  Database Management [icon: database, color: green]
}

Quality Assurance [color: orange, icon: check-circle] {
  Test Dataset Curation [icon: folder, color: orange]
  LLM Accuracy Validation [icon: target, color: orange]
}

Systems Optimization [color: purple, icon: settings] {
  VRAM Management [icon: hard-drive, color: purple]
  Model Quantization [icon: sliders, color: purple]
  Local Deployment [icon: box, color: purple]
}

Technical Writing [color: red, icon: file-text] {
  Thesis Report [icon: book, color: red]
  Deadline Management [icon: calendar, color: red]
  Presentations [icon: airplay, color: red]
}

Project Delivery [shape: oval, color: lightgreen, icon: check]

// Relationships
Project Start > Backend Development
Project Start > UI Development
Project Start > Quality Assurance
Project Start > Systems Optimization
Project Start > Technical Writing

Backend Development > Project Delivery
UI Development > Project Delivery
Quality Assurance > Project Delivery
Systems Optimization > Project Delivery
Technical Writing > Project Delivery