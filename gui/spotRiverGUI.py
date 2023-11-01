import tkinter as tk
from tkinter import ttk

from spotRiver.tuner.run import run_spot_river_experiment, compare_tuned_default

result = None
fun_control = None

def on_core_model_select(event):
    selected_core_model = core_model_var.get()
    if selected_core_model == "AMFClassifier":
        # If "AMFClassifier" is selected, update the options to include both models
        core_model_menu['menu'].delete(0, 'end')
        core_model_options = ["AMFClassifier", "HoeffdingAdaptiveTreeClassifier"]
        for option in core_model_options:
            core_model_menu['menu'].add_command(label=option, command=tk._setit(core_model_var, option))

    elif selected_core_model == "HoeffdingAdaptiveTreeClassifier":
        # If "HoeffdingAdaptiveTreeClassifier" is selected, update the options accordingly
        core_model_menu['menu'].delete(0, 'end')
        core_model_options = ["HoeffdingAdaptiveTreeClassifier", "AMFClassifier"]
        for option in core_model_options:
            core_model_menu['menu'].add_command(label=option, command=tk._setit(core_model_var, option))



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
    prep_model = prep_model_var.get()
    core_model = core_model_var.get()

    result, fun_control = run_spot_river_experiment(
        MAX_TIME=MAX_TIME,
        INIT_SIZE=INIT_SIZE,
        PREFIX=PREFIX,
        horizon=horizon,
        n_samples=n_samples,
        n_train=n_train,
        oml_grace_period=oml_grace_period,
        data_set=data_set,
        prepmodel=prep_model,
        coremodel=core_model,
    )

def analyze_data():
    if result is not None and fun_control is not None and compare_tuned_default_var.get():
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

prep_model_label = ttk.Label(run_tab, text="Select preprocessing model")
prep_model_label.grid(row=9, column=0, sticky="W")
prep_model_var = tk.StringVar()
prep_model_var.set("StandardScaler")  # Default selection
prep_model_options = ["StandardScaler", "MinMaxScaler", "None"]
prep_model_menu = ttk.OptionMenu(run_tab, prep_model_var, *prep_model_options)
prep_model_menu.grid(row=9, column=1, sticky="W")


core_model_label = ttk.Label(run_tab, text="Select core model")
core_model_label.grid(row=10, column=0, sticky="W")
core_model_var = tk.StringVar()
core_model_var.set("AMFClassifier")  # Default selection
core_model_options = ["AMFClassifier", "HoeffdingAdaptiveTreeClassifier"]
core_model_menu = ttk.OptionMenu(run_tab, core_model_var, *core_model_options, command=on_core_model_select)
core_model_menu.grid(row=10, column=1, columnspan=2, sticky="W")


run_button = ttk.Button(run_tab, text="Run Experiment", command=run_experiment)
run_button.grid(row=11, column=3, columnspan=2, sticky="E")

# Create and pack the "Analysis" tab with a button to run the analysis
analysis_tab = ttk.Frame(notebook)
notebook.add(analysis_tab, text="Analysis")

notebook.pack()


# Add the Logo image in both tabs
logo_image = tk.PhotoImage(file="images/spotlogo.png")
logo_label = tk.Label(run_tab, image=logo_image)
logo_label.grid(row=0, column=6, rowspan=1, columnspan=1)

analysis_label = tk.Label(analysis_tab, text="Analysis options:")
analysis_label.grid(row=0, column=1, sticky="W")

compare_tuned_default_var = tk.BooleanVar(value=True)
compare_tuned_default_checkbox = tk.Checkbutton(analysis_tab, text="Compare tuned vs. default", variable=compare_tuned_default_var)
compare_tuned_default_checkbox.grid(row=2, column=1, sticky="W")

analyze_button = ttk.Button(analysis_tab, text="Analyze Data", command=analyze_data)
analyze_button.grid(row=3, column=2, columnspan=2, sticky="E")

logo_label = tk.Label(analysis_tab, image=logo_image)
logo_label.grid(row=0, column=6, rowspan=1, columnspan=1)

# Run the mainloop

app.mainloop()
