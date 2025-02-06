import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.interpolate import CubicSpline, interp1d
from scipy.optimize import curve_fit
import sympy as sp

# Main Application Class
class MathApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Numerical Methods Calculator")
        self.root.geometry("900x700")
        self.default_tol = 1e-6
        
        # Main container frame
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Top Frame for method selection
        self.top_frame = ttk.Frame(self.main_frame)
        self.top_frame.pack(fill=tk.X, pady=5)

        self.method_label = ttk.Label(self.top_frame, text="Select Method:")
        self.method_label.pack(side=tk.LEFT)

        self.methods = [
            "Absolute/Relative Errors",
            "Graphical Method",
            "Bisection Method",
            "Newton-Raphson",
            "Relaxation Method",
            "Power Method",
            "Exponential Fit",
            "Spline Interpolation",
            "Picard’s Method",
            "Simpson’s Rule"
        ]
        self.method_combo = ttk.Combobox(self.top_frame, values=self.methods, width=30, state="readonly")
        self.method_combo.set("Select a method")
        self.method_combo.pack(side=tk.LEFT, padx=10)
        self.method_combo.bind("<<ComboboxSelected>>", self.update_inputs)

        # Separator Frame for input parameters and outputs
        self.mid_frame = ttk.Frame(self.main_frame)
        self.mid_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Left: Input fields container inside a Labelframe
        self.input_frame = ttk.LabelFrame(self.mid_frame, text="Parameters", padding="10")
        self.input_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Right: Output and Plot area in a notebook
        self.output_frame = ttk.Notebook(self.mid_frame)
        self.output_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Tab for textual results
        self.text_tab = ttk.Frame(self.output_frame)
        self.output_frame.add(self.text_tab, text="Results")
        self.result_text = tk.Text(self.text_tab, height=10, width=50, state='disabled', wrap=tk.WORD)
        self.result_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Tab for plots/graphs
        self.plot_tab = ttk.Frame(self.output_frame)
        self.output_frame.add(self.plot_tab, text="Graph/Plot")
        self.figure = plt.Figure(figsize=(6, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_tab)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Bottom Frame for buttons
        self.btn_frame = ttk.Frame(self.main_frame)
        self.btn_frame.pack(fill=tk.X, pady=10)

        self.execute_btn = ttk.Button(self.btn_frame, text="Calculate", command=self.execute_method)
        self.execute_btn.pack(side=tk.LEFT, padx=5)
        self.clear_btn = ttk.Button(self.btn_frame, text="Clear All", command=self.clear_all)
        self.clear_btn.pack(side=tk.LEFT, padx=5)

        # Message label in input area (shown when no method is selected)
        self.placeholder_label = ttk.Label(self.input_frame, text="Select a method to display parameters.", foreground="gray")
        self.placeholder_label.pack(pady=20)

        self.current_widgets = []

    def update_inputs(self, event=None):
        # Clear previous input widgets
        for widget in self.current_widgets:
            widget.destroy()
        self.current_widgets = []
        # Remove placeholder
        self.placeholder_label.destroy()

        method = self.method_combo.get()
        inputs = []

        if method == "Absolute/Relative Errors":
            inputs = [
                ("True Value:", "entry"),
                ("Approximate Value:", "entry")
            ]
        elif method == "Graphical Method":
            inputs = [
                ("Function (e.g., x**4 - 10*x**2 + 9):", "entry"),
                ("Start X:", "entry"),
                ("End X:", "entry")
            ]
        elif method == "Bisection Method":
            inputs = [
                ("Function (use 'np' for math, e.g., x**3 - 6*x**2 + 11*x - 6):", "entry"),
                ("Left boundary (a):", "entry"),
                ("Right boundary (b):", "entry"),
                ("Tolerance (optional):", "entry")
            ]
        elif method == "Newton-Raphson":
            inputs = [
                ("Function (e.g., x**3 - 2*x - 5):", "entry"),
                ("Derivative (e.g., 3*x**2 - 2):", "entry"),
                ("Initial guess:", "entry"),
                ("Tolerance (optional):", "entry")
            ]
        elif method == "Relaxation Method":
            inputs = [
                ("Matrix A (rows separated by ';' and numbers by commas, e.g., '10,-1,2,0; -1,11,-1,3; 2,-1,10,-1; 0,3,-1,8'):", "entry"),
                ("Vector b (comma separated, e.g., '6,25,-11,15'):", "entry"),
                ("Initial guess (comma separated, e.g., '0,0,0,0'):", "entry"),
                ("Omega:", "entry"),
                ("Tolerance (optional):", "entry"),
                ("Max iterations (optional):", "entry")
            ]
        elif method == "Power Method":
            inputs = [
                ("Matrix A (rows separated by ';'):", "entry"),
                ("Initial vector (comma separated):", "entry"),
                ("Tolerance (optional):", "entry"),
                ("Max iterations (optional):", "entry")
            ]
        elif method == "Exponential Fit":
            inputs = [
                ("X values (comma separated):", "entry"),
                ("Y values (comma separated):", "entry")
            ]
        elif method == "Spline Interpolation":
            inputs = [
                ("X values (comma separated):", "entry"),
                ("Y values (comma separated):", "entry"),
                ("X to interpolate:", "entry"),
                ("Type (cubic/quadratic):", "entry")
            ]
        elif method == "Picard’s Method":
            inputs = [
                ("ODE (as a function of x and y, e.g., x + y):", "entry"),
                ("Initial y0:", "entry"),
                ("Number of iterations:", "entry"),
                ("x value to evaluate:", "entry")
            ]
        elif method == "Simpson’s Rule":
            inputs = [
                ("Function (e.g., np.sin(x)):", "entry"),
                ("Start (a):", "entry"),
                ("End (b):", "entry"),
                ("Subintervals (even number):", "entry")
            ]

        # Create input fields
        for i, (label_text, widget_type) in enumerate(inputs):
            lbl = ttk.Label(self.input_frame, text=label_text)
            lbl.grid(row=i, column=0, sticky=tk.W, padx=5, pady=2)
            entry = ttk.Entry(self.input_frame, width=40)
            entry.grid(row=i, column=1, sticky=tk.EW, padx=5, pady=2)
            self.current_widgets.extend([lbl, entry])

    def get_input_values(self):
        entries = [w for w in self.input_frame.winfo_children() if isinstance(w, ttk.Entry)]
        return [e.get() for e in entries]

    def execute_method(self):
        method = self.method_combo.get()
        inputs = self.get_input_values()

        self.clear_text()
        self.figure.clf()

        try:
            if method == "Absolute/Relative Errors":
                self.handle_absolute_errors(inputs)
            elif method == "Graphical Method":
                self.handle_graphical_method(inputs)
            elif method == "Bisection Method":
                self.handle_bisection(inputs)
            elif method == "Newton-Raphson":
                self.handle_newton_raphson(inputs)
            elif method == "Relaxation Method":
                self.handle_relaxation(inputs)
            elif method == "Power Method":
                self.handle_power_method(inputs)
            elif method == "Exponential Fit":
                self.handle_exponential_fit(inputs)
            elif method == "Spline Interpolation":
                self.handle_spline(inputs)
            elif method == "Picard’s Method":
                self.handle_picard(inputs)
            elif method == "Simpson’s Rule":
                self.handle_simpson(inputs)
            else:
                self.append_text("Select a valid method.")
            self.canvas.draw()
        except Exception as e:
            self.append_text(f"Error: {str(e)}")
            messagebox.showerror("Error", str(e))

    def append_text(self, text):
        self.result_text.configure(state='normal')
        self.result_text.insert(tk.END, text + "\n")
        self.result_text.configure(state='disabled')

    def clear_text(self):
        self.result_text.configure(state='normal')
        self.result_text.delete(1.0, tk.END)
        self.result_text.configure(state='disabled')

    # -------------------------------
    # Method Implementations
    # -------------------------------

    def handle_absolute_errors(self, inputs):
        if not inputs[0] or not inputs[1]:
            raise ValueError("Both True Value and Approximate Value are required.")
        true_val = float(inputs[0])
        approx_val = float(inputs[1])
        abs_err = abs(true_val - approx_val)
        rel_err = abs_err / true_val if true_val != 0 else float('inf')
        self.append_text(f"Absolute Error: {abs_err:.6f}")
        self.append_text(f"Relative Error: {rel_err:.6%}")

    def handle_graphical_method(self, inputs):
        if not all(inputs[:3]):
            raise ValueError("All fields are required for Graphical Method.")
        # Using eval safely with np and x in namespace
        f = lambda x: eval(inputs[0], {"np": np, "x": x})
        start = float(inputs[1])
        end = float(inputs[2])
        x = np.linspace(start, end, 400)
        y = f(x)
        ax = self.figure.add_subplot(111)
        ax.plot(x, y, label=f"f(x) = {inputs[0]}")
        ax.axhline(0, color='red', linestyle='--', label="y=0")
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.legend()
        ax.grid()
        self.append_text("Graph plotted.")

    def handle_bisection(self, inputs):
        if not all(inputs[:3]):
            raise ValueError("Function and boundaries a, b are required.")
        f = lambda x: eval(inputs[0], {"np": np, "x": x})
        a = float(inputs[1])
        b = float(inputs[2])
        tol = float(inputs[3]) if inputs[3] else self.default_tol

        if f(a) * f(b) >= 0:
            raise ValueError("f(a) and f(b) must have opposite signs.")
        iteration = 0
        while (b - a)/2 > tol:
            c = (a + b)/2
            if f(c) == 0:
                a = b = c
                break
            if f(a) * f(c) < 0:
                b = c
            else:
                a = c
            iteration += 1
        root = (a + b)/2
        self.append_text(f"Approximate root: {root:.6f} (in {iteration} iterations)")

    def handle_newton_raphson(self, inputs):
        if not all(inputs[:3]):
            raise ValueError("Function, derivative, and initial guess are required.")
        f = lambda x: eval(inputs[0], {"np": np, "x": x})
        df = lambda x: eval(inputs[1], {"np": np, "x": x})
        x0 = float(inputs[2])
        tol = float(inputs[3]) if inputs[3] else self.default_tol
        iteration = 0
        x = x0
        while abs(f(x)) > tol:
            x = x - f(x) / df(x)
            iteration += 1
            if iteration > 1000:
                raise RuntimeError("Newton-Raphson did not converge.")
        self.append_text(f"Root: {x:.6f} (in {iteration} iterations)")

    def handle_relaxation(self, inputs):
        # Expected inputs: Matrix A, Vector b, Initial guess, Omega, Tol, Max iter.
        if not all(inputs[:4]):
            raise ValueError("Matrix A, Vector b, Initial guess, and Omega are required.")
        try:
            A = np.array([[float(num) for num in row.split(',')] for row in inputs[0].split(';')])
            b = np.array([float(num) for num in inputs[1].split(',')])
            x0 = np.array([float(num) for num in inputs[2].split(',')])
        except Exception as e:
            raise ValueError("Error parsing matrix or vectors. Ensure proper format.")
        omega = float(inputs[3])
        tol = float(inputs[4]) if len(inputs) > 4 and inputs[4] else self.default_tol
        max_iter = int(inputs[5]) if len(inputs) > 5 and inputs[5] else 100

        n = len(b)
        x = x0.copy()
        iteration = 0
        for _ in range(max_iter):
            x_new = x.copy()
            for i in range(n):
                summation = sum(A[i][j] * x_new[j] for j in range(n) if j != i)
                if A[i][i] == 0:
                    raise ValueError("Diagonal can not be zero")
                x_new[i] = (1 - omega) * x[i] + omega * (b[i] - summation) / A[i][i]
            iteration += 1
            if np.linalg.norm(x_new - x, ord=np.inf) < tol:
                self.append_text(f"Solution: {np.round(x_new, 6)} (in {iteration} iterations)")
                return
            x = x_new
        self.append_text(f"Approximate solution after {max_iter} iterations: {np.round(x, 6)}")

    def handle_power_method(self, inputs):
        if not all(inputs[:2]):
            raise ValueError("Matrix A and Initial vector are required.")
        try:
            A = np.array([[float(num) for num in row.split(',')] for row in inputs[0].split(';')])
            v0 = np.array([float(num) for num in inputs[1].split(',')])
        except Exception as e:
            raise ValueError("Error parsing matrix or vector. Check the format.")
        tol = float(inputs[2]) if len(inputs) > 2 and inputs[2] else self.default_tol
        max_iter = int(inputs[3]) if len(inputs) > 3 and inputs[3] else 100
        # Normalize initial vector
        v = v0 / np.linalg.norm(v0)
        iteration = 0
        for _ in range(max_iter):
            w = np.dot(A, v)
            lambda_new = np.dot(w, v)
            v_new = w / np.linalg.norm(w)
            iteration += 1
            if np.linalg.norm(v_new - v) < tol:
                self.append_text(f"Largest eigenvalue: {lambda_new:.6f}")
                self.append_text(f"Corresponding eigenvector: {v_new}")
                self.append_text(f"Iterations: {iteration}")
                return
            v = v_new
        self.append_text(f"Approximate largest eigenvalue: {lambda_new:.6f} (after {max_iter} iterations)")

    def handle_exponential_fit(self, inputs):
        if not all(inputs[:2]):
            raise ValueError("X and Y values are required.")
        try:
            x_vals = np.array([float(num) for num in inputs[0].split(',')])
            y_vals = np.array([float(num) for num in inputs[1].split(',')])
        except Exception as e:
            raise ValueError("Error parsing X or Y values.")
        # Exponential model: y = a * exp(b*x)
        def exp_func(x, a, b):
            return a * np.exp(b * x)
        params, _ = curve_fit(exp_func, x_vals, y_vals)
        a, b = params
        self.append_text(f"Fitted model: y = {a:.6f} * exp({b:.6f} * x)")
        # Plot the original data and fitted curve
        x_fit = np.linspace(min(x_vals), max(x_vals), 200)
        y_fit = exp_func(x_fit, a, b)
        ax = self.figure.add_subplot(111)
        ax.scatter(x_vals, y_vals, color='orange', label="Data Points")
        ax.plot(x_fit, y_fit, color='red', label="Exponential Fit")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Exponential Curve Fitting")
        ax.legend()
        ax.grid()

    def handle_spline(self, inputs):
        if not all(inputs[:3]):
            raise ValueError("X values, Y values, and X to interpolate are required.")
        try:
            x_vals = np.array([float(num) for num in inputs[0].split(',')])
            y_vals = np.array([float(num) for num in inputs[1].split(',')])
            x_interp = float(inputs[2])
        except Exception as e:
            raise ValueError("Error parsing input values for spline interpolation.")
        spline_type = inputs[3].strip().lower() if len(inputs) > 3 else "cubic"
        if spline_type == "cubic":
            cs = CubicSpline(x_vals, y_vals)
            y_interp = cs(x_interp)
        elif spline_type == "quadratic":
            spline_func = interp1d(x_vals, y_vals, kind='quadratic')
            y_interp = spline_func(x_interp)
        else:
            raise ValueError("Spline type must be 'cubic' or 'quadratic'.")
        self.append_text(f"Interpolated value at x = {x_interp}: {y_interp:.6f}")
        # Also plot the spline
        x_dense = np.linspace(min(x_vals), max(x_vals), 400)
        if spline_type == "cubic":
            y_dense = CubicSpline(x_vals, y_vals)(x_dense)
        else:
            y_dense = interp1d(x_vals, y_vals, kind='quadratic')(x_dense)
        ax = self.figure.add_subplot(111)
        ax.plot(x_dense, y_dense, label=f"{spline_type.capitalize()} Spline")
        ax.scatter(x_vals, y_vals, color='orange', label="Data Points")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Spline Interpolation")
        ax.legend()
        ax.grid()

    def handle_picard(self, inputs):
        if not all(inputs[:4]):
            raise ValueError("ODE, initial y0, number of iterations, and x value are required.")
        ode_expr = inputs[0]
        y0 = float(inputs[1])
        iterations = int(inputs[2])
        x_val = float(inputs[3])
        # Define symbolic variables
        x_sym = sp.symbols('x')
        y = sp.Function('y')(x_sym)
        # Parse the ODE function f(x,y)
        f_expr = sp.sympify(ode_expr)
        # Picard's iteration: y_{n+1}(x) = y0 + ∫[0,x] f(t, y_n(t)) dt
        # Start with the zeroth approximation (constant function)
        y_approx = sp.sympify(y0)
        approximations = []
        for i in range(iterations):
            t = sp.symbols('t')
            # Replace x by t and y by previous approximation
            f_sub = f_expr.subs({x_sym: t, y: y_approx})
            y_new = y0 + sp.integrate(f_sub, (t, 0, x_sym))
            approximations.append(sp.simplify(y_new))
            y_approx = y_new
        # Evaluate the last approximation at x = x_val
        y_val = y_approx.subs(x_sym, x_val)
        self.append_text(f"After {iterations} iterations, y({x_val}) = {y_val.evalf()}")
        # Optionally, list all approximations:
        for idx, approx in enumerate(approximations, start=1):
            self.append_text(f"Iteration {idx}: y(x) = {sp.pretty(approx)}")

    def handle_simpson(self, inputs):
        if not all(inputs[:4]):
            raise ValueError("Function, start, end, and number of subintervals are required.")
        f = lambda x: eval(inputs[0], {"np": np, "x": x})
        a = float(inputs[1])
        b = float(inputs[2])
        n = int(inputs[3])
        if n % 2 != 0:
            raise ValueError("Subintervals must be an even number.")
        x = np.linspace(a, b, n + 1)
        y = f(x)
        h = (b - a) / n
        I = h/3 * (y[0] + y[-1] + 4 * sum(y[1:-1:2]) + 2 * sum(y[2:-2:2]))
        self.append_text(f"Approximate integral value: {I:.6f}")

    # -------------------------------
    # Clear function
    def clear_all(self):
        # Clear inputs
        for widget in self.input_frame.winfo_children():
            widget.destroy()
        self.current_widgets = []
        # Reset placeholder message
        self.placeholder_label = ttk.Label(self.input_frame, text="Select a method to display parameters.", foreground="gray")
        self.placeholder_label.pack(pady=20)
        # Clear text and figure
        self.clear_text()
        self.figure.clf()
        self.canvas.draw()

# Main execution
if __name__ == "__main__":
    root = tk.Tk()
    app = MathApp(root)
    root.mainloop()
