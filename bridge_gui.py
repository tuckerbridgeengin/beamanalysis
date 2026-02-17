import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, messagebox
import re 

# ==========================================
#  CORE ANALYSIS ENGINE
# ==========================================
class BridgeBeam:
    def __init__(self, L_ft, E_ksi, supports_ft, num_elements=200):
        self.L = L_ft
        self.E = E_ksi
        self.supports = np.array(supports_ft)
        self.n_elem = num_elements
        self.nodes = np.linspace(0, L_ft, num_elements + 1)
        self.dx = L_ft / num_elements
        self.n_nodes = len(self.nodes)
        self.n_dof = 2 * self.n_nodes
        
        # Results Containers
        self.results = {
            'x': self.nodes,
            'max_disp': np.zeros(self.n_nodes),   
            'min_disp': np.zeros(self.n_nodes),
            'max_moment': np.zeros(self.n_nodes), 
            'min_moment': np.zeros(self.n_nodes),
            'max_shear': np.zeros(self.n_nodes),  
            'min_shear': np.zeros(self.n_nodes),
            'max_reaction': {x: 0.0 for x in supports_ft}, 
            'min_reaction': {x: 0.0 for x in supports_ft}
        }
        
        self.results['max_disp'][:] = -np.inf
        self.results['max_moment'][:] = -np.inf
        self.results['max_shear'][:] = -np.inf
        self.results['min_disp'][:] = np.inf
        self.results['min_moment'][:] = np.inf
        self.results['min_shear'][:] = np.inf
        
        self.I_func = None

    def set_stiffness_profile(self, I_in4_function):
        self.I_func = I_in4_function

    def _get_element_stiffness(self, x_start_ft, x_end_ft):
        # 1. Calculate Lengths
        L_ft = x_end_ft - x_start_ft
        L_in = L_ft * 12.0  # Convert to inches
        
        # 2. Get Inertia at Midpoint
        x_mid_ft = (x_start_ft + x_end_ft) / 2.0
        I_val = self.I_func(x_mid_ft)
        
        # 3. Define L for the matrix (The error was here previously)
        L = L_in 
        
        # 4. Construct Stiffness Matrix (Standard Beam Element)
        k = (self.E * I_val / L**3) * np.array([
            [12, 6*L, -12, 6*L],
            [6*L, 4*L**2, -6*L, 2*L**2],
            [-12, -6*L, 12, -6*L],
            [6*L, 2*L**2, -6*L, 4*L**2]
        ])
        return k

    def _build_global_stiffness(self):
        K = np.zeros((self.n_dof, self.n_dof))
        for i in range(self.n_elem):
            x0 = self.nodes[i]
            x1 = self.nodes[i+1]
            k_el = self._get_element_stiffness(x0, x1)
            idx = [2*i, 2*i+1, 2*(i+1), 2*(i+1)+1]
            for r in range(4):
                for c in range(4):
                    K[idx[r], idx[c]] += k_el[r, c]
        return K

    def analyze_truck(self, truck_config):
        if self.I_func is None:
            raise ValueError("Stiffness profile not defined.")

        K_global = self._build_global_stiffness()

        fixed_dofs = []
        support_indices = []
        for sup_x in self.supports:
            idx = (np.abs(self.nodes - sup_x)).argmin()
            fixed_dofs.append(2 * idx) 
            support_indices.append(idx)
            
        free_dofs = [i for i in range(self.n_dof) if i not in fixed_dofs]
        K_ff = K_global[np.ix_(free_dofs, free_dofs)]
        
        try:
            K_inv = np.linalg.inv(K_ff)
        except np.linalg.LinAlgError:
            messagebox.showerror("Error", "Structure is unstable! Check supports.")
            return

        truck_length = max([axle[1] for axle in truck_config])
        start_pos = -truck_length
        end_pos = self.L + self.dx
        
        truck_positions = np.arange(start_pos, end_pos, self.dx)
        
        for i_step, front_axle_x in enumerate(truck_positions):
            F_global = np.zeros(self.n_dof)
            is_truck_on_bridge = False
            
            for load, offset in truck_config:
                axle_x = front_axle_x - offset
                if 0 <= axle_x <= self.L:
                    is_truck_on_bridge = True
                    elem_idx = int(axle_x // self.dx)
                    if elem_idx >= self.n_elem: elem_idx = self.n_elem - 1
                    
                    x_left = self.nodes[elem_idx]
                    x_right = self.nodes[elem_idx + 1]
                    dist_into_elem = axle_x - x_left
                    element_len = x_right - x_left
                    ratio_right = dist_into_elem / element_len
                    ratio_left = 1.0 - ratio_right
                    
                    F_global[2 * elem_idx] -= load * ratio_left
                    F_global[2 * (elem_idx + 1)] -= load * ratio_right
            
            if not is_truck_on_bridge: continue

            F_free = F_global[free_dofs]
            u_free = K_inv @ F_free
            u_total = np.zeros(self.n_dof)
            u_total[free_dofs] = u_free
            
            disp_y = u_total[0::2]
            R_total = K_global @ u_total - F_global
            
            for sup_idx, sup_x in zip(support_indices, self.supports):
                r_val = R_total[2 * sup_idx]
                if self.results['max_reaction'][sup_x] == 0 and self.results['min_reaction'][sup_x] == 0:
                    self.results['max_reaction'][sup_x] = r_val
                    self.results['min_reaction'][sup_x] = r_val
                else:
                    self.results['max_reaction'][sup_x] = max(self.results['max_reaction'][sup_x], r_val)
                    self.results['min_reaction'][sup_x] = min(self.results['min_reaction'][sup_x], r_val)

            moments = np.zeros(self.n_nodes)
            shears = np.zeros(self.n_nodes)
            
            for j in range(self.n_elem):
                idx = [2*j, 2*j+1, 2*(j+1), 2*(j+1)+1]
                u_el = u_total[idx]
                k_el = self._get_element_stiffness(self.nodes[j], self.nodes[j+1])
                f_el = k_el @ u_el
                
                shears[j] = f_el[0]
                moments[j] = -f_el[1] / 12.0 
                if j == self.n_elem - 1:
                    shears[j+1] = -f_el[2] 
                    moments[j+1] = f_el[3] / 12.0

            self.results['max_disp'] = np.maximum(self.results['max_disp'], disp_y)
            self.results['min_disp'] = np.minimum(self.results['min_disp'], disp_y)
            self.results['max_moment'] = np.maximum(self.results['max_moment'], moments)
            self.results['min_moment'] = np.minimum(self.results['min_moment'], moments)
            self.results['max_shear'] = np.maximum(self.results['max_shear'], shears)
            self.results['min_shear'] = np.minimum(self.results['min_shear'], shears)

    def plot_envelopes(self):
        x = self.nodes
        fig, axs = plt.subplots(4, 1, figsize=(10, 16), sharex=False, 
                                gridspec_kw={'height_ratios': [3, 3, 3, 1]})
        
        axs[0].sharex(axs[1])
        axs[1].sharex(axs[2])

        for i in range(3):
            for sx in self.supports:
                axs[i].axvline(x=sx, color='k', linestyle=':', alpha=0.5)

        axs[0].fill_between(x, self.results['max_disp'], self.results['min_disp'], color='blue', alpha=0.1)
        axs[0].plot(x, self.results['max_disp'], 'b-', label='Max')
        axs[0].plot(x, self.results['min_disp'], 'b-', label='Min')
        axs[0].set_title("Deflection Envelope (in)")
        axs[0].set_ylabel("Deflection (in)")
        axs[0].invert_yaxis()
        axs[0].grid(True)
        plt.setp(axs[0].get_xticklabels(), visible=False)

        axs[1].fill_between(x, self.results['max_shear'], self.results['min_shear'], color='green', alpha=0.1)
        axs[1].plot(x, self.results['max_shear'], 'g--')
        axs[1].plot(x, self.results['min_shear'], 'g--')
        axs[1].set_title("Shear Force Envelope (kips)")
        axs[1].set_ylabel("Shear (kips)")
        axs[1].grid(True)
        plt.setp(axs[1].get_xticklabels(), visible=False)

        axs[2].fill_between(x, self.results['max_moment'], self.results['min_moment'], color='red', alpha=0.1)
        axs[2].plot(x, self.results['max_moment'], 'r--')
        axs[2].plot(x, self.results['min_moment'], 'r--')
        axs[2].set_title("Bending Moment Envelope (kip-ft)")
        axs[2].set_ylabel("Moment (kip-ft)")
        axs[2].set_xlabel("Distance along Bridge (ft)")
        axs[2].grid(True)

        axs[3].axis('off')
        axs[3].set_title("Support Reactions Envelope", pad=20)
        
        columns = ["Support Location (ft)", "Max Reaction (kips)", "Min Reaction (kips)"]
        cell_text = []
        for pos in self.supports:
            max_r = self.results['max_reaction'][pos]
            min_r = self.results['min_reaction'][pos]
            cell_text.append([f"{pos:.1f}", f"{max_r:.2f}", f"{min_r:.2f}"])

        table = axs[3].table(cellText=cell_text, colLabels=columns, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2.0)

        plt.tight_layout()
        plt.show()

# ==========================================
#  GUI INTERFACE
# ==========================================

class BridgeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Bridge Analysis Tool")
        self.root.geometry("600x800")

        # --- 1. Geometry Section ---
        lbl_frame_geo = ttk.LabelFrame(root, text="Geometry & Material")
        lbl_frame_geo.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(lbl_frame_geo, text="Modulus E [ksi]:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.ent_E = ttk.Entry(lbl_frame_geo)
        self.ent_E.insert(0, "3600")
        self.ent_E.grid(row=0, column=1, padx=5, pady=2)

        ttk.Label(lbl_frame_geo, text="Supports [ft] (comma sep):").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        self.ent_supports = ttk.Entry(lbl_frame_geo)
        self.ent_supports.insert(0, "0, 100, 200")
        self.ent_supports.grid(row=1, column=1, padx=5, pady=2)
        
        ttk.Label(lbl_frame_geo, text="(Total Length will be the last support location)").grid(row=2, column=0, columnspan=2, sticky="w", padx=5)

        # --- 2. Stiffness Section ---
        lbl_frame_stiff = ttk.LabelFrame(root, text="Beam Stiffness (Moment of Inertia)")
        lbl_frame_stiff.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(lbl_frame_stiff, text="Define I [in^4] segments (one per line):").pack(anchor="w", padx=5)
        ttk.Label(lbl_frame_stiff, text="Format: 'VALUE from START to END'", font=("Arial", 8, "italic")).pack(anchor="w", padx=5)
        
        self.txt_stiff = tk.Text(lbl_frame_stiff, height=6, width=60)
        self.txt_stiff.pack(padx=5, pady=5)
        
        # Default Example
        default_stiff = "50000 from 0 to 75\n120000 from 75 to 125\n50000 from 125 to end"
        self.txt_stiff.insert("1.0", default_stiff)

        # --- 3. Truck Load Section ---
        lbl_frame_load = ttk.LabelFrame(root, text="Truck Configuration")
        lbl_frame_load.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(lbl_frame_load, text="Define Axles (Weight [kips], Dist from Front [ft])").pack(anchor="w", padx=5)
        self.txt_truck = tk.Text(lbl_frame_load, height=5, width=60)
        self.txt_truck.pack(padx=5, pady=5)
        self.txt_truck.insert("1.0", "8.0, 0.0\n32.0, 14.0\n32.0, 28.0")

        # --- Run Button ---
        btn_run = ttk.Button(root, text="Run Analysis", command=self.run_analysis)
        btn_run.pack(pady=20, ipadx=10, ipady=5)
        
        self.status_lbl = ttk.Label(root, text="Ready.")
        self.status_lbl.pack()

    def parse_stiffness_text(self, text, total_length):
        segments = []
        lines = text.strip().split('\n')
        pattern = re.compile(r"([\d\.]+).*?([\d\.]+).*?([\d\.]+|end)", re.IGNORECASE)
        
        for line in lines:
            if not line.strip(): continue
            match = pattern.search(line)
            if match:
                val = float(match.group(1))
                start = float(match.group(2))
                end_str = match.group(3).lower()
                end = total_length if end_str == 'end' else float(end_str)
                segments.append((start, end, val))
            else:
                try:
                    parts = [x.strip() for x in line.split(',')]
                    if len(parts) == 3:
                        end_val = total_length if parts[2].lower() == 'end' else float(parts[2])
                        segments.append((float(parts[1]), end_val, float(parts[0])))
                except:
                    print(f"Skipping invalid line: {line}")
        return segments

    def run_analysis(self):
        try:
            self.status_lbl.config(text="Processing inputs...")
            self.root.update()

            E = float(self.ent_E.get())
            
            sup_str = self.ent_supports.get()
            supports = sorted([float(x.strip()) for x in sup_str.split(',')])
            if not supports:
                messagebox.showerror("Error", "Please enter at least one support.")
                return

            # Auto Length = Last Support
            L = supports[-1]
            if L <= 0:
                messagebox.showerror("Error", "Bridge length must be greater than zero.")
                return

            truck_str = self.txt_truck.get("1.0", "end-1c").strip()
            truck_config = []
            for line in truck_str.split('\n'):
                if not line.strip(): continue
                parts = line.split(',')
                if len(parts) == 2:
                    truck_config.append((float(parts[0]), float(parts[1])))

            if not truck_config:
                messagebox.showerror("Error", "Please define at least one axle.")
                return

            stiff_text = self.txt_stiff.get("1.0", "end-1c")
            segments = self.parse_stiffness_text(stiff_text, L)
            
            if not segments:
                I_default = 50000.0
                segments = [(0, L, I_default)]

            def get_I(x):
                for (start, end, val) in segments:
                    if start <= x <= end:
                        return val
                return segments[0][2] 

            self.status_lbl.config(text=f"Analyzing {L}ft Bridge...")
            self.root.update()
            
            bridge = BridgeBeam(L, E, supports, num_elements=300)
            bridge.set_stiffness_profile(get_I)
            bridge.analyze_truck(truck_config)
            
            self.status_lbl.config(text="Plotting results...")
            bridge.plot_envelopes()
            self.status_lbl.config(text="Analysis Complete.")
            
        except ValueError:
            messagebox.showerror("Input Error", "Check your numbers.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = BridgeApp(root)
    root.mainloop()