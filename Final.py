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
        self.root.geometry("1200x800")
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
                ("True Value (e.g., 3.14159):", "entry"),
                ("Approximate Value (e.g., 3.14):", "entry")
            ]
        elif method == "Graphical Method":
            inputs = [
                ("Function (e.g., x**4 - 10*x**2 + 9):", "entry"),
                ("Start X (e.g., -4):", "entry"),
                ("End X (e.g., 4):", "entry")
            ]
        elif method == "Bisection Method":
            inputs = [
                ("Function (use 'np' for math, e.g., x**3 - 6*x**2 + 11*x - 6):", "entry"),
                ("Left boundary (a) (e.g., 0):", "entry"),
                ("Right boundary (b) (e.g., 3):", "entry"),
                ("Tolerance (optional, e.g., 1e-6):", "entry")
            ]
        elif method == "Newton-Raphson":
            inputs = [
                ("Function (e.g., x**3 - 2*x - 5):", "entry"),
                ("Derivative (auto-filled):", "entry"),
                ("Initial guess (e.g., 2.5):", "entry"),
                ("Tolerance (optional, e.g., 1e-6):", "entry")
            ]
        elif method == "Relaxation Method":
            inputs = [
                ("Matrix A (rows separated by ';', numbers by commas, e.g., '1,1,1;1,0,1;0,1,1'):", "entry"),
                ("Vector b (comma separated, e.g., '10,6,8'):", "entry"),
                ("Initial guess (comma separated, e.g., '0,0,0'):", "entry"),
                ("Omega (e.g., 0.8):", "entry"),
                ("Tolerance (optional, e.g., 1e-6):", "entry"),
                ("Max iterations (optional, e.g., 100):", "entry")
            ]
        elif method == "Power Method":
            inputs = [
                ("Matrix A (rows separated by ';', e.g., '4,1;2,3'):", "entry"),
                ("Initial vector (comma separated, e.g., '1,1'):", "entry"),
                ("Tolerance (optional, e.g., 1e-6):", "entry"),
                ("Max iterations (optional, e.g., 100):", "entry")
            ]
        elif method == "Exponential Fit":
            inputs = [
                ("X values (comma separated, e.g., '0,1,2,3,4'):", "entry"),
                ("Y values (comma separated, e.g., '2.5,3.5,7.4,20.5,54.6'):", "entry")
            ]
        elif method == "Spline Interpolation":
            inputs = [
                ("X values (comma separated, e.g., '0,0.5,1.0,1.5'):", "entry"),
                ("Y values (comma separated, e.g., '0,0.25,0.75,2.25'):", "entry"),
                ("X to interpolate (e.g., 1.5):", "entry"),
                ("Type (cubic/quadratic):", "entry")
            ]
        elif method == "Picard’s Method":
            inputs = [
                ("ODE (as a function of x and y, e.g., x + y):", "entry"),
                ("Initial y0 (e.g., 1):", "entry"),
                ("Number of iterations (e.g., 4):", "entry"),
                ("x value to evaluate (e.g., 0.2):", "entry")
            ]
        elif method == "Simpson’s Rule":
            inputs = [
                ("Function (e.g., np.sin(x)):", "entry"),
                ("Start (a) (e.g., 0):", "entry"),
                ("End (b) (e.g., np.pi):", "entry"),
                ("Subintervals (even number, e.g., 10):", "entry")
            ]

        # Create input fields
        for i, (label_text, widget_type) in enumerate(inputs):
            lbl = ttk.Label(self.input_frame, text=label_text)
            lbl.grid(row=i, column=0, sticky=tk.W, padx=5, pady=2)
            entry = ttk.Entry(self.input_frame, width=40)
            entry.grid(row=i, column=1, sticky=tk.EW, padx=5, pady=2)
            self.current_widgets.extend([lbl, entry])

        if method == "Newton-Raphson":
            # Automatically add derivative field and make it read-only
            func_entry = self.current_widgets[1]
            func_entry.bind("<FocusOut>", self.update_derivative)

    def get_input_values(self):
        entries = [w for w in self.input_frame.winfo_children() if isinstance(w, ttk.Entry)]
        return [e.get() for e in entries]

    def update_derivative(self, event):
        func_entry = event.widget
        func_text = func_entry.get()
        x = sp.symbols('x')
        func = sp.sympify(func_text)
        derivative = str(sp.diff(func, x))
        # Update the derivative entry field
        self.current_widgets[3].delete(0, tk.END)
        self.current_widgets[3].insert(0, derivative)
        self.current_widgets[3].configure(state='readonly')

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
        self.append_text(f"Relative Error: {rel_err:.6f}")

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
        
        midpoint = (a + b) / 2
        while abs(f(midpoint)) > tol:
            if f(a) * f(midpoint) < 0:
                b = midpoint
            else:
                a = midpoint
            midpoint = (a + b) / 2

        self.append_text(f"Approximate root: {midpoint:.6f}")

    def handle_newton_raphson(self, inputs):
        if not all(inputs[:3]):
            raise ValueError("Function, derivative, and initial guess are required.")
        f = lambda x: eval(inputs[0], {"np": np, "x": x})
        df = lambda x: eval(self.current_widgets[3].get(), {"np": np, "x": x})  # Use the auto-generated derivative
        x0 = float(inputs[2])
        tol = float(inputs[3]) if len(inputs) > 3 and inputs[3] else self.default_tol
        iteration = 0
        x = x0
        while abs(f(x)) > tol:
            x = x - f(x) / df(x)
            iteration += 1
            if iteration > 1000:
                raise RuntimeError("Newton-Raphson did not converge.")
        self.append_text(f"Root: {x:.6f} (in {iteration} iterations)")

    def handle_relaxation(self, inputs):
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

        # Check for zero diagonals and swap rows if needed
        for i in range(len(A)):
            if A[i, i] == 0:
                for j in range(i + 1, len(A)):
                    if A[j, i] != 0 and A[i, j] != 0:
                        A[[i, j]] = A[[j, i]]
                        b[[i, j]] = b[[j, i]]
                        break
                else:
                    raise ValueError("Cannot solve the system: Zero diagonal element found and no suitable row to swap.")

        n = len(b)
        x = x0.copy()
        iteration = 0
        for _ in range(max_iter):
            x_new = x.copy()
            for i in range(n):
                summation = sum(A[i][j] * x_new[j] for j in range(n) if j != i)
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
        def exp_func(x, a, b):
            return a * np.exp(b * x)
        params, _ = curve_fit(exp_func, x_vals, y_vals)
        a, b = params
        self.append_text(f"Fitted model: y = {a:.6f} * exp({b:.6f} * x)")
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
        x_dense = np.linspace(min(x_vals), max(x_vals), 400)
        if spline_type == "cubic":
            y_dense = CubicSpline(x_vals, y_vals)(x_dense)
        else:
            y_dense = interp1d(x_vals, y_vals, kind='quadratic')(x_dense)
        ax = self.figure.add_subplot(111)
        ax.plot(x_dense, y_dense, label=f"Spline ({spline_type})")
        ax.scatter(x_vals, y_vals, color='orange', label="Data Points")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend()
        ax.grid()

    def handle_picard(self, inputs):
        if not all(inputs[:4]):
            raise ValueError("ODE, Initial y0, Number of iterations, and x value to evaluate are required.")
        ode = lambda x, y: eval(inputs[0], {"np": np, "x": x, "y": y})
        y0 = float(inputs[1])
        iter_count = int(inputs[2])
        x_val = float(inputs[3])
        y = y0
        for _ in range(iter_count):
            y = y0 + ode(x_val, y)
        self.append_text(f"Picard's approximation at x = {x_val}: {y:.6f}")

    def handle_simpson(self, inputs):
        if not all(inputs[:4]):
            raise ValueError("Function, Start (a), End (b), and Subintervals are required.")
        f = lambda x: eval(inputs[0], {"np": np, "x": x})
        a = float(inputs[1])
        b = float(inputs[2])
        n = int(inputs[3])
        if n % 2 != 0:
            raise ValueError("Number of subintervals must be even.")
        h = (b - a) / n
        integral = f(a) + f(b)
        for i in range(1, n, 2):
            integral += 4 * f(a + i * h)
        for i in range(2, n-1, 2):
            integral += 2 * f(a + i * h)
        integral *= h / 3
        self.append_text(f"Simpson's rule approximation: {integral:.6f}")

    def clear_all(self):
        self.clear_text()
        self.figure.clf()
        self.canvas.draw()
        self.method_combo.set("Select a method")
        for widget in self.current_widgets:
            widget.destroy()
        self.current_widgets = []
        self.placeholder_label = ttk.Label(self.input_frame, text="Select a method to display parameters.", foreground="gray")
        self.placeholder_label.pack(pady=20)

if __name__ == "__main__":
    root = tk.Tk()
    app = MathApp(root)
    root.mainloop()