import numpy as np
import matplotlib.pyplot as plt


def synthesize_bottom_bc(
    action,  # shape (2*modes,) -> [a1,b1,a2,b2,...,aN,bN]
    modes,
    actuator_limit,  # fraction of Δb allowed for global amplitude
    Lx=2 * np.pi,  # domain length in x (your sim uses L=[2π, 2])
    Δb=1.0,  # bottom-top temperature difference at bottom (default from your sim)
    T0=2.0,  # base bottom temperature = min_b + Δb (min_b=1, Δb=1 -> T0=2)
    W=512,  # number of x-samples for plotting
):
    action = np.asarray(action, dtype=float)
    assert action.shape[0] == 2 * modes, (
        f"Expected action of length {2 * modes}, got {action.shape[0]}"
    )
    x = np.linspace(0.0, Lx, W, endpoint=False)

    # raw fluctuation s(x) = sum_n a_n cos + b_n sin
    s = np.zeros_like(x)
    for n in range(1, modes + 1):
        a = action[2 * n - 2]
        b = action[2 * n - 1]
        θ = 2 * np.pi * n * x / Lx
        s += a * np.cos(θ) + b * np.sin(θ)

    # amplitude safeguard (match Julia): scale by K = max(1, sumabs / (actuator_limit * Δb))
    sumabs = np.abs(action).sum()
    K = max(1.0, sumabs / (actuator_limit * Δb))
    s_safe = s / K

    Tb = T0 + s_safe  # final bottom boundary temperature
    # Per-mode contributions after the same scaling (useful for plotting)
    mode_contribs = []
    for n in range(1, modes + 1):
        a = action[2 * n - 2]
        b = action[2 * n - 1]
        θ = 2 * np.pi * n * x / Lx
        mode_contribs.append((a * np.cos(θ) + b * np.sin(θ)) / K)

    return x, Tb, s_safe, mode_contribs


def plot_control(
    action, modes, actuator_limit, Lx=2 * np.pi, Δb=1.0, T0=2.0, W=512, show_modes=True
):
    x, Tb, s_safe, mode_contribs = synthesize_bottom_bc(
        action, modes, actuator_limit, Lx, Δb, T0, W
    )

    # Plot resulting boundary temperature
    plt.figure()
    plt.plot(x, Tb, label="Bottom boundary T(x)")
    plt.axhline(T0, linestyle="--", label="Base T0")
    plt.xlabel("x")
    plt.ylabel("T at y=0")
    plt.title("Synthesized bottom boundary temperature")
    plt.legend()
    plt.tight_layout()

    if show_modes:
        # Plot the zero-mean fluctuation and its per-mode contributions
        plt.figure()
        plt.plot(x, s_safe, label="Total fluctuation s(x)")
        for i, mc in enumerate(mode_contribs, start=1):
            plt.plot(x, mc, label=f"mode n={i}")
        plt.xlabel("x")
        plt.ylabel("fluctuation")
        plt.title("Mode contributions (post-scaling)")
        plt.legend(ncol=2)
        plt.tight_layout()


# ============================
# Live rollout visualization
# ============================
class ControlPlotter:
    """
    Persistent, live-updating plotter for the Fourier control.
    Creates a single figure with two axes:
      - Top: bottom boundary temperature T_b(x)
      - Bottom: zero-mean fluctuation and individual mode contributions
    Reuse the same artists between updates to avoid spawning new figures.
    """
    def __init__(
        self,
        modes,
        actuator_limit,
        Lx=2 * np.pi,
        Δb=1.0,
        T0=2.0,
        W=96,
        show_modes=True,
        max_modes=None,       # cap number of per-mode lines shown (None = show all)
    ):
        self.modes = modes
        self.actuator_limit = actuator_limit
        self.Lx = Lx
        self.Δb = Δb
        self.T0 = T0
        self.W = W
        self.show_modes = show_modes
        self.max_modes = max_modes if max_modes is not None else modes

        # Precompute x grid and create figure + axes
        self.x = np.linspace(0.0, Lx, W, endpoint=False)
        self.fig, (self.ax0, self.ax1) = plt.subplots(
            2, 1, figsize=(8, 6), sharex=True
        )
        self.fig.canvas.manager.set_window_title("Fourier Control (live)")

        # Top axis: T_b(x)
        (self.line_T,) = self.ax0.plot(self.x, np.full_like(self.x, T0), label="T_b(x)")
        (self.line_T0,) = self.ax0.plot(self.x, np.full_like(self.x, T0), linestyle="--", label="Base T0")
        self.ax0.set_ylabel("T at y=0")
        self.ax0.set_title("Bottom boundary temperature (live)")
        self.ax0.legend(loc="upper right")
        # Set stable y-limits around expected range
        lim = 1.1 * actuator_limit * Δb
        self.ax0.set_ylim(T0 - lim, T0 + lim)

        # Bottom axis: fluctuation + modes
        (self.line_total,) = self.ax1.plot(self.x, np.zeros_like(self.x), label="Total fluctuation s(x)")
        self.mode_lines = []
        modes_to_draw = min(self.max_modes, modes)
        for i in range(modes_to_draw):
            (line_i,) = self.ax1.plot(self.x, np.zeros_like(self.x), label=f"mode n={i+1}")
            self.mode_lines.append(line_i)
        self.ax1.set_xlabel("x")
        self.ax1.set_ylabel("fluctuation")
        self.ax1.set_title("Mode contributions (post-scaling)")
        self.ax1.legend(ncol=2, loc="upper right")
        self.ax1.set_ylim(-lim, lim)

        # Interactive mode for live updates
        plt.ion()
        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update(self, action):
        """
        Update the plots in-place using a new action vector (length 2*modes).
        """
        x, Tb, s_safe, mode_contribs = synthesize_bottom_bc(
            action,
            modes=self.modes,
            actuator_limit=self.actuator_limit,
            Lx=self.Lx,
            Δb=self.Δb,
            T0=self.T0,
            W=self.W,
        )
        # Reuse precomputed x to avoid re-lining
        self.line_T.set_ydata(Tb)
        self.line_T0.set_ydata(np.full_like(self.x, self.T0))
        self.line_total.set_ydata(s_safe)

        # Update mode lines (respect max_modes cap)
        for i, line in enumerate(self.mode_lines):
            if i < len(mode_contribs):
                line.set_ydata(mode_contribs[i])
            else:
                line.set_ydata(np.zeros_like(self.x))

        # Efficient redraw
        self.fig.canvas.draw_idle()
        plt.pause(0.001)

def start_live_control(
    modes,
    actuator_limit,
    Lx=2 * np.pi,
    Δb=1.0,
    T0=2.0,
    W=512,
    show_modes=True,
    max_modes=None,
):
    """
    Convenience constructor to start a live plotter.
    Returns a ControlPlotter instance.
    """
    return ControlPlotter(
        modes=modes,
        actuator_limit=actuator_limit,
        Lx=Lx,
        Δb=Δb,
        T0=T0,
        W=W,
        show_modes=show_modes,
        max_modes=max_modes,
    )

def update_live_control(plotter: "ControlPlotter", action):
    """
    Convenience function to update a live plotter with a new action vector.
    """
    plotter.update(action)
