import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd
import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors, Lipinski, QED
from PIL import Image, ImageTk
import os
import threading
import queue
import time
from sklearn.preprocessing import StandardScaler
import sys

def resource_path(relative_path):
    """Get absolute path to resource, works in dev and in PyInstaller .exe"""
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

class MLScanLCKApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MLScan-LCK: Machine Learning-based drug Scanning for LCK inhibition")

        # Responsive design
        self.root.geometry("1280x800")
        self.root.minsize(1080, 700)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Queue for threads
        self.batch_queue = queue.Queue()
        self.resources = {
            "models": {},
            "feature_indices": None,
            "scaler": None
        }

        # Color palette (changed for LCK version)
        self.bg_color = '#EDF6F9'
        self.header_color = '#2E4053'   # <-- CHANGE HEADER COLOR HERE
        self.footer_color = '#117864'   # <-- NEW FOOTER COLOR
        self.accent_color = '#E63946'
        self.secondary_color = '#FCA311'
        self.success_color = '#40916C'
        self.warning_color = '#FFB703'
        self.error_color = '#D90429'
        self.text_color = '#14213D'
        self.light_color = '#BDE0FE'

        self.root.configure(bg=self.bg_color)

        # Style configuration
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('TFrame', background=self.bg_color)
        self.style.configure('TLabel', background=self.bg_color, foreground=self.text_color, font=('Montserrat', 10))
        self.style.configure('TButton', font=('Montserrat', 10, 'bold'), padding=8)
        self.style.configure('Accent.TButton', background=self.accent_color, foreground='white')
        self.style.map('Accent.TButton', background=[('active', self.secondary_color)])
        self.style.configure('Secondary.TButton', background=self.secondary_color, foreground='white')
        self.style.map('Secondary.TButton', background=[('active', self.accent_color)])
        self.style.configure('Header.TLabel', font=('Montserrat', 18, 'bold'), foreground='white', background=self.header_color)
        self.style.configure('Subheader.TLabel', font=('Montserrat', 12), foreground='white', background=self.header_color)
        self.style.configure('TLabelframe', background=self.bg_color, foreground=self.text_color, font=('Montserrat', 11, 'bold'))
        self.style.configure('TLabelframe.Label', background=self.bg_color, foreground=self.text_color)
        self.style.configure("Custom.Horizontal.TProgressbar",
                             troughcolor=self.light_color,
                             background=self.success_color,
                             thickness=20)
        self.style.configure("Treeview",
                             font=('Montserrat', 9),
                             rowheight=26,
                             background="#ffffff",
                             fieldbackground="#ffffff",
                             bordercolor="#dee2e6")
        self.style.configure("Treeview.Heading",
                             font=('Montserrat', 9, 'bold'),
                             background=self.light_color,
                             relief=tk.FLAT)
        self.style.map("Treeview.Heading", background=[('active', self.secondary_color)])

        # Header Frame
        header_frame = tk.Frame(root, bg=self.header_color)  # color changed here
        header_frame.pack(fill=tk.X)
        tk.Label(header_frame, text="MLScan-LCK", font=('Montserrat', 22, 'bold'), fg='white', bg=self.header_color).pack(pady=(10,0))
        tk.Label(header_frame, text="Machine Learning-based drug Scanning for LCK inhibition", font=('Montserrat', 13), fg='white', bg=self.header_color).pack(pady=(0,10))

        # Main container
        self.main_container = tk.Frame(root, bg=self.bg_color)
        self.main_container.pack(fill=tk.BOTH, expand=True)

        # Notebook (tabs)
        self.notebook = ttk.Notebook(self.main_container)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        self.tab1 = ttk.Frame(self.notebook, padding=10)
        self.tab2 = ttk.Frame(self.notebook, padding=10)

        self.notebook.add(self.tab1, text="Single Molecule")
        self.notebook.add(self.tab2, text="Batch Processing")

        self.load_resources()
        self.create_single_molecule_tab()
        self.create_batch_processing_tab()
        self.create_footer()

        # Queue processor
        self.root.after(100, self.process_queue)
        self.root.bind('<Configure>', self.sticky_footer)
        self.root.bind('<Map>', self.sticky_footer)
        self.notebook.bind('<<NotebookTabChanged>>', self.sticky_footer)

    def load_resources(self):
        try:
            self.resources["models"] = {}
            self.resources["feature_indices"] = None
            self.resources["scaler"] = None

            feature_file = resource_path("selected_feature_indices.npy")
            if os.path.exists(feature_file):
                self.resources["feature_indices"] = np.load(feature_file)
                if len(self.resources["feature_indices"]) != 100:
                    messagebox.showerror("Error", "Feature indices: Expected 100 features")
                    self.resources["feature_indices"] = None

            model_files = {
                "Random Forest": "RF.pkl",
                "SVM": "SVM.pkl",
                "KNN": "KNN.pkl",
                "XGBoost": "XGBoost.pkl",
                "LightGBM": "LightGBM.pkl",
                "Extra Trees": "ET.pkl"
            }
            for name, file in model_files.items():
                model_path = resource_path(file)
                if os.path.exists(model_path):
                    self.resources["models"][name] = joblib.load(model_path)

            scaler_path = resource_path("scaler.pkl")
            if os.path.exists(scaler_path):
                self.resources["scaler"] = joblib.load(scaler_path)
        except Exception as e:
            messagebox.showerror("Error", f"Error loading resources: {str(e)}")

    def create_single_molecule_tab(self):
        main_frame = tk.Frame(self.tab1, bg=self.bg_color)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Use a PanedWindow to allocate more space and allow side-by-side layout
        pw = tk.PanedWindow(main_frame, orient=tk.HORIZONTAL, sashrelief=tk.RAISED, bg=self.bg_color)
        pw.pack(fill=tk.BOTH, expand=True)

        left_panel = tk.Frame(pw, bg=self.bg_color)
        right_panel = tk.Frame(pw, bg=self.bg_color)
        pw.add(left_panel, minsize=420)
        pw.add(right_panel, minsize=420)

        # Input frame
        input_frame = ttk.LabelFrame(left_panel, text=" SMILES Input ", padding=12)
        input_frame.pack(fill=tk.X, pady=7)
        self.smiles_entry = ttk.Entry(input_frame, width=54, font=('Montserrat', 10))
        self.smiles_entry.pack(fill=tk.X, pady=5)
        self.smiles_entry.insert(0, "CN1C=NC2=C1C(=O)N(C(=O)N2C)C")

        # Model selection: model checkboxes in a single row (side by side)
        model_frame = ttk.LabelFrame(left_panel, text=" Model Selection ", padding=12)
        model_frame.pack(fill=tk.X, pady=7)
        self.model_vars = {}

        model_row = tk.Frame(model_frame, bg=self.bg_color)
        model_row.pack(fill=tk.X)
        for i, model_name in enumerate(self.resources["models"].keys()):
            var = tk.BooleanVar(value=True)
            cb = ttk.Checkbutton(model_row, text=model_name, variable=var)
            cb.pack(side=tk.LEFT, padx=12, pady=2)
            self.model_vars[model_name] = var

        analyze_btn = ttk.Button(left_panel, text="Analyze Molecule", command=self.analyze_single_molecule, style='Accent.TButton')
        analyze_btn.pack(pady=12, fill=tk.X)

        # Molecular structure: increased area (width and height)
        mol_frame = ttk.LabelFrame(left_panel, text=" Molecular Structure ", padding=12)
        mol_frame.pack(fill=tk.BOTH, expand=True, pady=7)
        # Larger canvas for molecular structure: 560x360
        self.mol_canvas = tk.Canvas(mol_frame, width=560, height=360, bg='white', highlightthickness=0, bd=0)
        self.mol_canvas.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self.mol_canvas.bind('<Configure>', self.redraw_molecule)

        self.current_smiles = None
        self.current_mol_img = None

        props_frame = ttk.LabelFrame(right_panel, text=" Molecular Properties ", padding=12)
        props_frame.pack(fill=tk.BOTH, expand=True, pady=7)
        self.props_text = scrolledtext.ScrolledText(
            props_frame, height=11, wrap=tk.WORD, font=('Consolas', 10),
            padx=6, pady=6, bg='white', bd=1, relief=tk.SOLID
        )
        self.props_text.pack(fill=tk.BOTH, expand=True)

        preds_frame = ttk.LabelFrame(right_panel, text=" Model Predictions ", padding=12)
        preds_frame.pack(fill=tk.BOTH, expand=True, pady=7)
        self.preds_text = scrolledtext.ScrolledText(
            preds_frame, height=14, wrap=tk.WORD, font=('Consolas', 10),
            padx=6, pady=6, bg='white', bd=1, relief=tk.SOLID
        )
        self.preds_text.pack(fill=tk.BOTH, expand=True)

    def analyze_single_molecule(self):
        smiles = self.smiles_entry.get().strip()
        if not smiles:
            messagebox.showwarning("Warning", "Please enter a SMILES string")
            return
        selected_models = [name for name, var in self.model_vars.items() if var.get()]
        if not selected_models:
            messagebox.showwarning("Warning", "Please select at least one model")
            return
        self.props_text.delete(1.0, tk.END)
        self.preds_text.delete(1.0, tk.END)
        if not self.display_molecule(smiles):
            messagebox.showerror("Error", "Invalid SMILES string")
            return
        props = self.calculate_properties(smiles)
        if not props:
            messagebox.showerror("Error", "Could not calculate properties")
            return

        props_text = f"""Molecular Weight: {props['MW']:.3f} {"(≤500)" if props['MW'] <= 500 else "(>500)"}
LogP: {props['LogP']:.3f} {"(≤5)" if props['LogP'] <= 5 else "(>5)"}
H-bond Acceptors: {props['HBA']} {"(≤10)" if props['HBA'] <= 10 else "(>10)"}
H-bond Donors: {props['HBD']} {"(≤5)" if props['HBD'] <= 5 else "(>5)"}
Rotatable Bonds: {props['RotBonds']}
TPSA: {props['TPSA']:.3f}
QED: {props['QED']:.3f} (Weight: {props['QED_Weight']:.3f}, LogP: {props['QED_LogP']:.3f}, Size: {props['QED_Size']:.3f})
QED Components: HBA: {props['QED_HBA']:.3f}, HBD: {props['QED_HBD']:.3f}, Polar: {props['QED_Polar']:.3f}
QED Components: Insaturation: {props['QED_Insaturation']:.3f}, RotBonds: {props['QED_RotBonds']:.3f}\n"""

        self.props_text.insert(tk.END, props_text)
        if props["Lipinski_Violations"] == 0:
            self.props_text.insert(tk.END, "\nZero Lipinski violations (good drug-like properties)", "success")
        else:
            self.props_text.insert(tk.END, f"\n{props['Lipinski_Violations']} Lipinski violation(s)", "warning")
        self.props_text.tag_config("success", foreground=self.success_color)
        self.props_text.tag_config("warning", foreground=self.warning_color)

        features = self.smiles_to_features(smiles, self.resources["feature_indices"])
        if features is None:
            messagebox.showerror("Error", "Could not generate valid features for this molecule")
            return

        for name in selected_models:
            model = self.resources["models"].get(name)
            if model is None:
                continue
            try:
                X = features.reshape(1, -1)
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X)[0][1]
                else:
                    proba = float(model.predict(X)[0])
                pred = "Active" if proba >= 0.5 else "Inactive"
                color = "success" if pred == "Active" else "error"
                self.preds_text.insert(tk.END, f"{name}:\t", "bold")
                self.preds_text.insert(tk.END, f"{pred}\t", color)
                self.preds_text.insert(tk.END, f"Probability: {proba:.3f}\n")
            except Exception as e:
                self.preds_text.insert(tk.END, f"{name}:\tError in prediction: {str(e)}\n", "error")
        self.preds_text.tag_config("bold", font=('Montserrat', 10, 'bold'))
        self.preds_text.tag_config("success", foreground=self.success_color)
        self.preds_text.tag_config("error", foreground=self.error_color)

    def redraw_molecule(self, event):
        """Redraw molecule structure image when canvas is resized."""
        if self.current_smiles:
            self.display_molecule(self.current_smiles, force_redraw=True)

    def display_molecule(self, smiles, force_redraw=False):
        """Display molecule structure, always fitting inside canvas."""
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return False
        self.current_smiles = smiles
        # Get current canvas size
        width = self.mol_canvas.winfo_width()
        height = self.mol_canvas.winfo_height()
        # Minimum image size
        width = max(width, 320)
        height = max(height, 230)
        img = Draw.MolToImage(mol, size=(int(width * 0.98), int(height * 0.98)))
        self.current_mol_img = ImageTk.PhotoImage(img)
        self.mol_canvas.delete("all")
        self.mol_canvas.create_image(
            width / 2,
            height / 2,
            anchor=tk.CENTER,
            image=self.current_mol_img
        )
        return True

    def create_batch_processing_tab(self):
        main_frame = tk.Frame(self.tab2, bg=self.bg_color)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        top_panel = tk.Frame(main_frame, bg=self.bg_color)
        top_panel.pack(fill=tk.X, pady=6)
        middle_panel = tk.Frame(main_frame, bg=self.bg_color)
        middle_panel.pack(fill=tk.X, pady=6)
        bottom_panel = tk.Frame(main_frame, bg=self.bg_color)
        bottom_panel.pack(fill=tk.BOTH, expand=True, pady=6)

        file_frame = ttk.LabelFrame(top_panel, text=" CSV File Input ", padding=12)
        file_frame.pack(fill=tk.X)
        file_input_frame = tk.Frame(file_frame, bg=self.bg_color)
        file_input_frame.pack(fill=tk.X)
        self.file_entry = ttk.Entry(file_input_frame, font=('Montserrat', 10))
        self.file_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        browse_btn = ttk.Button(file_input_frame, text="Browse", command=self.browse_file, style='Secondary.TButton')
        browse_btn.pack(side=tk.LEFT, padx=5)

        # Model selection in batch: models side by side in a single row
        model_frame = ttk.LabelFrame(top_panel, text=" Model Selection ", padding=12)
        model_frame.pack(fill=tk.X, pady=7)
        self.batch_model_vars = {}
        batch_model_row = tk.Frame(model_frame, bg=self.bg_color)
        batch_model_row.pack(fill=tk.X)
        for i, model_name in enumerate(self.resources["models"].keys()):
            var = tk.BooleanVar(value=True)
            cb = ttk.Checkbutton(batch_model_row, text=model_name, variable=var)
            cb.pack(side=tk.LEFT, padx=12, pady=2)
            self.batch_model_vars[model_name] = var

        button_frame = tk.Frame(top_panel, bg=self.bg_color)
        button_frame.pack(pady=12)
        self.process_btn = ttk.Button(button_frame, text="Process Batch", command=self.start_batch_processing, style='Accent.TButton')
        self.process_btn.pack(side=tk.LEFT, padx=6)
        self.stop_btn = ttk.Button(button_frame, text="Stop", command=self.stop_batch_processing, state=tk.DISABLED, style='Secondary.TButton')
        self.stop_btn.pack(side=tk.LEFT, padx=6)
        self.download_btn = ttk.Button(button_frame, text="Download Results", command=self.download_results, style='Accent.TButton', state=tk.DISABLED)
        self.download_btn.pack(side=tk.LEFT, padx=6)

        progress_frame = ttk.LabelFrame(middle_panel, text=" Processing Progress ", padding=12)
        progress_frame.pack(fill=tk.X)
        self.progress_label = ttk.Label(progress_frame, text="Ready", font=('Montserrat', 10))
        self.progress_label.pack(anchor=tk.W)
        self.progress_count = ttk.Label(progress_frame, text="0/0 compounds processed", font=('Montserrat', 9))
        self.progress_count.pack(anchor=tk.W)
        self.progress_bar = ttk.Progressbar(progress_frame, orient=tk.HORIZONTAL,
                                            mode='determinate', style="Custom.Horizontal.TProgressbar")
        self.progress_bar.pack(fill=tk.X, pady=6)

        results_frame = ttk.LabelFrame(bottom_panel, text=" Results ", padding=12)
        results_frame.pack(fill=tk.BOTH, expand=True)
        tree_container = tk.Frame(results_frame, bg=self.bg_color)
        tree_container.pack(fill=tk.BOTH, expand=True)
        self.results_tree = ttk.Treeview(tree_container, selectmode='extended')
        yscroll = ttk.Scrollbar(tree_container, orient=tk.VERTICAL, command=self.results_tree.yview)
        yscroll.pack(side=tk.RIGHT, fill=tk.Y)
        xscroll = ttk.Scrollbar(tree_container, orient=tk.HORIZONTAL, command=self.results_tree.xview)
        xscroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.results_tree.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        self.results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.results_tree.bind('<Control-c>', self.copy_selected_rows_to_clipboard)
        self.batch_status = ttk.Label(bottom_panel, text="", font=('Montserrat', 9))
        self.batch_status.pack()
        self.batch_results_df = None

    def copy_selected_rows_to_clipboard(self, event=None):
        selected_items = self.results_tree.selection()
        if not selected_items:
            return
        columns = self.results_tree["columns"]
        rows = []
        rows.append('\t'.join(columns))
        for item in selected_items:
            values = self.results_tree.item(item)['values']
            row_str = '\t'.join(str(v) for v in values)
            rows.append(row_str)
        clipboard_text = '\n'.join(rows)
        self.root.clipboard_clear()
        self.root.clipboard_append(clipboard_text)

    def display_results_table(self, df):
        self.results_tree.delete(*self.results_tree.get_children())
        numeric_cols = ['MW', 'LogP', 'TPSA', 'QED', 'QED_Weight', 'QED_LogP',
                        'QED_Size', 'QED_HBA', 'QED_HBD', 'QED_Polar',
                        'QED_Insaturation', 'QED_RotBonds']
        prob_cols = [col for col in df.columns if '_Prob' in col]
        numeric_cols.extend(prob_cols)
        formatted_df = df.copy()
        for col in numeric_cols:
            if col in formatted_df.columns:
                formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.3f}" if pd.notnull(x) else "")
        columns = list(formatted_df.columns)
        self.results_tree["columns"] = columns
        col_widths = {
            "SMILES": 150,
            "MW": 80,
            "LogP": 80,
            "HBA": 60,
            "HBD": 60,
            "RotBonds": 80,
            "TPSA": 80,
            "Lipinski_Violations": 120,
            "QED": 80,
            "QED_Weight": 100,
            "QED_LogP": 100,
            "QED_Size": 100,
            "QED_HBA": 100,
            "QED_HBD": 100,
            "QED_Polar": 100,
            "QED_Insaturation": 120,
            "QED_RotBonds": 120
        }
        for col in columns:
            if "_Pred" in col:
                col_widths[col] = 100
            elif "_Prob" in col:
                col_widths[col] = 120
        for col in columns:
            self.results_tree.heading(col, text=col)
            width = col_widths.get(col, 100)
            self.results_tree.column(col, width=width, anchor=tk.CENTER, stretch=False)
        for i, row in formatted_df.iterrows():
            tag = 'evenrow' if i % 2 == 0 else 'oddrow'
            self.results_tree.insert("", tk.END, values=list(row), tags=(tag,))
        self.results_tree.tag_configure('evenrow', background='#EDF6F9')
        self.results_tree.tag_configure('oddrow', background='#ffffff')
        self.batch_results_df = df
        self.download_btn.config(state=tk.NORMAL)

    def create_footer(self):
        self.footer_frame = tk.Frame(self.root, bg=self.footer_color, height=38)  # color changed here
        self.footer_frame.place(relx=0, rely=1.0, relwidth=1.0, anchor='sw')
        self.footer_frame.pack_propagate(False)
        self.dev_label = tk.Label(
            self.footer_frame,
            text="Developed by: Sheikh Sunzid Ahmed & M. Oliur Rahman",
            font=("Montserrat", 10, "bold"),
            fg='white',
            bg=self.footer_color
        )
        self.dev_label.pack(side=tk.LEFT, padx=18, anchor="w")
        self.dept_label = tk.Label(
            self.footer_frame,
            text="Plant Taxonomy and Ethnobotany Laboratory, University of Dhaka",
            font=("Montserrat", 10),
            fg='white',
            bg=self.footer_color
        )
        self.dept_label.pack(side=tk.RIGHT, padx=18, anchor="e")

    def sticky_footer(self, event=None):
        self.footer_frame.place(relx=0, rely=1.0, relwidth=1.0, anchor='sw')
        self.footer_frame.lift()
        self.footer_frame.update_idletasks()

    def browse_file(self):
        filepath = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if filepath:
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, filepath)

    def smiles_to_features(self, smiles, feature_indices=None):
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        features = np.array(fp)
        if feature_indices is not None:
            try:
                features = features[feature_indices]
                if len(features) != 100:
                    messagebox.showerror("Error", f"Expected 100 features, got {len(features)}")
                    return None
            except IndexError as e:
                messagebox.showerror("Error", f"Feature selection error: {str(e)}")
                return None
        return features

    def calculate_properties(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return None
        qed_properties = QED.properties(mol)
        return {
            "MW": Descriptors.MolWt(mol),
            "LogP": Descriptors.MolLogP(mol),
            "HBA": Lipinski.NumHAcceptors(mol),
            "HBD": Lipinski.NumHDonors(mol),
            "RotBonds": Lipinski.NumRotatableBonds(mol),
            "TPSA": Descriptors.TPSA(mol),
            "Lipinski_Violations": sum([
                Descriptors.MolWt(mol) > 500,
                Descriptors.MolLogP(mol) > 5,
                Lipinski.NumHAcceptors(mol) > 10,
                Lipinski.NumHDonors(mol) > 5
            ]),
            "QED": QED.qed(mol),
            "QED_Weight": qed_properties[0],
            "QED_LogP": qed_properties[1],
            "QED_Size": qed_properties[2],
            "QED_HBA": qed_properties[3],
            "QED_HBD": qed_properties[4],
            "QED_Polar": qed_properties[5],
            "QED_Insaturation": qed_properties[6],
            "QED_RotBonds": qed_properties[7]
        }

    def start_batch_processing(self):
        filepath = self.file_entry.get().strip()
        if not filepath:
            messagebox.showwarning("Warning", "Please select a CSV file")
            return
        if not os.path.exists(filepath):
            messagebox.showerror("Error", "File does not exist")
            return
        if self.resources["feature_indices"] is None:
            messagebox.showerror("Error", "Cannot process batch - feature selection not available")
            return
        self.selected_batch_models = [name for name, var in self.batch_model_vars.items() if var.get()]
        if not self.selected_batch_models:
            messagebox.showwarning("Warning", "Please select at least one model")
            return
        try:
            self.df = pd.read_csv(filepath)
            if "SMILES" not in self.df.columns:
                messagebox.showerror("Error", "CSV must contain a 'SMILES' column")
                return
        except Exception as e:
            messagebox.showerror("Error", f"Could not read CSV file: {str(e)}")
            return
        self.stop_processing = False
        self.processed_count = 0
        self.total_count = len(self.df)
        self.results_tree.delete(*self.results_tree.get_children())
        self.batch_status.config(text="")
        self.process_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.download_btn.config(state=tk.DISABLED)
        self.progress_bar['value'] = 0
        self.progress_label.config(text="Processing...")
        self.progress_count.config(text=f"0/{self.total_count} compounds processed")
        threading.Thread(
            target=self.process_batch_thread,
            args=(self.df,),
            daemon=True
        ).start()

    def stop_batch_processing(self):
        self.stop_processing = True
        self.process_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.progress_label.config(text="Processing stopped by user")
        self.download_btn.config(state=tk.DISABLED)

    def process_batch_thread(self, df):
        results = []
        invalid_smiles = []
        error_molecules = []
        for i, smi in enumerate(df["SMILES"]):
            if self.stop_processing:
                break
            try:
                mol = Chem.MolFromSmiles(smi)
                if not mol:
                    invalid_smiles.append(i + 2)
                    continue
                props = self.calculate_properties(smi)
                if not props:
                    error_molecules.append((i + 2, "Property calculation failed"))
                    continue
                features = self.smiles_to_features(smi, self.resources["feature_indices"])
                if features is None:
                    error_molecules.append((i + 2, "Feature generation failed"))
                    continue
                preds = {"SMILES": smi}
                for name in self.selected_batch_models:
                    model = self.resources["models"].get(name)
                    if not model:
                        continue
                    try:
                        X = features.reshape(1, -1)
                        if hasattr(model, "predict_proba"):
                            proba = model.predict_proba(X)[0][1]
                        else:
                            proba = float(model.predict(X)[0])
                        preds[f"{name}_Pred"] = "Active" if proba >= 0.5 else "Inactive"
                        preds[f"{name}_Prob"] = proba
                    except Exception as e:
                        preds[f"{name}_Pred"] = "Error"
                        preds[f"{name}_Prob"] = np.nan
                        error_molecules.append((i + 2, f"{name} prediction error: {str(e)}"))
                preds.update({
                    "MW": props["MW"],
                    "LogP": props["LogP"],
                    "HBA": props["HBA"],
                    "HBD": props["HBD"],
                    "RotBonds": props["RotBonds"],
                    "TPSA": props["TPSA"],
                    "Lipinski_Violations": props["Lipinski_Violations"],
                    "QED": props["QED"],
                    "QED_Weight": props["QED_Weight"],
                    "QED_LogP": props["QED_LogP"],
                    "QED_Size": props["QED_Size"],
                    "QED_HBA": props["QED_HBA"],
                    "QED_HBD": props["QED_HBD"],
                    "QED_Polar": props["QED_Polar"],
                    "QED_Insaturation": props["QED_Insaturation"],
                    "QED_RotBonds": props["QED_RotBonds"]
                })
                results.append(preds)
            except Exception as e:
                error_molecules.append((i + 2, f"General processing error: {str(e)}"))
            self.processed_count = i + 1
            self.batch_queue.put(("progress", (i + 1, len(df))))
        self.batch_queue.put(("results", (results, invalid_smiles, error_molecules)))

    def process_queue(self):
        try:
            while True:
                try:
                    msg_type, data = self.batch_queue.get_nowait()
                    if msg_type == "progress":
                        processed, total = data
                        progress = (processed / total) * 100
                        self.progress_bar['value'] = progress
                        self.progress_count.config(text=f"{processed}/{total} compounds processed")
                        self.root.update_idletasks()
                    elif msg_type == "results":
                        results, invalid_smiles, error_molecules = data
                        if results:
                            self.results_df = pd.DataFrame(results)
                            self.display_results_table(self.results_df)
                            status_msg = f"Processed {len(results)} valid molecules"
                            if invalid_smiles:
                                status_msg += f" | Skipped {len(invalid_smiles)} invalid SMILES"
                            if error_molecules:
                                status_msg += f" | {len(error_molecules)} errors"
                            self.batch_status.config(text=status_msg)
                        else:
                            messagebox.showwarning("Warning", "No valid molecules processed")
                        self.process_btn.config(state=tk.NORMAL)
                        self.stop_btn.config(state=tk.DISABLED)
                        self.progress_label.config(text="Processing completed")
                except queue.Empty:
                    break
        finally:
            self.root.after(100, self.process_queue)

    def download_results(self):
        if not hasattr(self, 'results_df') or self.results_df.empty:
            messagebox.showwarning("Warning", "No results to download")
            return
        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
            title="Save Results As"
        )
        if filepath:
            try:
                self.results_df.to_csv(filepath, index=False)
                messagebox.showinfo("Success", f"Results saved to {filepath}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not save file: {str(e)}")

    def on_close(self):
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = MLScanLCKApp(root)
    root.mainloop()