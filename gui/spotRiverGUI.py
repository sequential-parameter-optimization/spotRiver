import tkinter as tk
from tkinter import ttk

from spotRiver.tuner.run import run_spot_river_experiment, compare_tuned_default

result = None
fun_control = None

def run_experiment():
    global result, fun_control
    MAX_TIME = float(max_time_entry.get())
    INIT_SIZE = int(init_size_entry.get())
    PREFIX = prefix_entry.get()
    horizon = int(horizon_entry.get())
    n_samples = int(n_samples_entry.get())
    n_train = int(n_train_entry.get())
    oml_grace_period = int(oml_grace_period_entry.get())
    data_set = data_set_var.get()

    result, fun_control = run_spot_river_experiment(
        MAX_TIME=MAX_TIME,
        INIT_SIZE=INIT_SIZE,
        PREFIX=PREFIX,
        horizon=horizon,
        n_samples=n_samples,
        n_train=n_train,
        oml_grace_period=oml_grace_period,
        data_set=data_set
    )

def analyze_data():
    if result is not None and fun_control is not None:
        compare_tuned_default(result, fun_control)  # Call the analysis method


# Create the main application window
app = tk.Tk()
app.title("Spot River Experiment GUI")

# Create a notebook (tabbed interface)
notebook = ttk.Notebook(app)
# notebook.pack(fill='both', expand=True)

# Create and pack entry fields for the "Run" tab
run_tab = ttk.Frame(notebook)
notebook.add(run_tab, text="Run")

max_time_label = tk.Label(run_tab, text="MAX_TIME:")
max_time_label.grid(row=1, column=0, sticky="W")
max_time_entry = tk.Entry(run_tab)
max_time_entry.insert(0, "1")
max_time_entry.grid(row=1, column=1)


init_size_label = tk.Label(run_tab, text="INIT_SIZE:")
init_size_label.grid(row=2, column=0, sticky="W")
init_size_entry = tk.Entry(run_tab)
init_size_entry.insert(0, "3")
init_size_entry.grid(row=2, column=1)

prefix_label = tk.Label(run_tab, text="PREFIX:")
prefix_label.grid(row=3, column=0, sticky="W")
prefix_entry = tk.Entry(run_tab)
prefix_entry.insert(0, "00")
prefix_entry.grid(row=3, column=1)

horizon_label = tk.Label(run_tab, text="horizon:")
horizon_label.grid(row=4, column=0, sticky="W")
horizon_entry = tk.Entry(run_tab)
horizon_entry.insert(0, "1")
horizon_entry.grid(row=4, column=1)

n_samples_label = tk.Label(run_tab, text="n_samples:")
n_samples_label.grid(row=5, column=0, sticky="W")
n_samples_entry = tk.Entry(run_tab)
n_samples_entry.insert(0, "1000")
n_samples_entry.grid(row=5, column=1)

n_train_label = tk.Label(run_tab, text="n_train:")
n_train_label.grid(row=6, column=0, sticky="W")
n_train_entry = tk.Entry(run_tab)
n_train_entry.insert(0, "100")
n_train_entry.grid(row=6, column=1)

oml_grace_period_label = tk.Label(run_tab, text="oml_grace_period:")
oml_grace_period_label.grid(row=7, column=0, sticky="W")
oml_grace_period_entry = tk.Entry(run_tab)
oml_grace_period_entry.insert(0, "100")
oml_grace_period_entry.grid(row=7, column=1)


data_set_label = ttk.Label(run_tab, text="Select data_set")
data_set_label.grid(row=8, column=0, sticky="W")
data_set_var = tk.StringVar()
data_set_var.set("Phishing")  # Default selection
data_set_options = ["Bananas", "CreditCard", "Phishing"]
data_set_menu = ttk.OptionMenu(run_tab, data_set_var, *data_set_options)
data_set_menu.grid(row=8, column=1, sticky="W")

run_button = ttk.Button(run_tab, text="Run Experiment", command=run_experiment)
run_button.grid(row=9, column=3, columnspan=2, sticky="E")

# Create and pack the "Analysis" tab with a button to run the analysis
analysis_tab = ttk.Frame(notebook)
notebook.add(analysis_tab, text="Analysis")

notebook.pack()

analyze_button = ttk.Button(analysis_tab, text="Analyze Data", command=analyze_data)
analyze_button.pack()

app.mainloop()
