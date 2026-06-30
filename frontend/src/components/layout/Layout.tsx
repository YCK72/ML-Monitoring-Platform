import { NavLink, Outlet } from "react-router-dom";
import { Activity, Bell, Gauge, LineChart, Search, Brain } from "lucide-react";

const navItems = [
  { to: "/", label: "Overview", icon: Gauge },
  { to: "/drift", label: "Drift Timeline", icon: LineChart },
  { to: "/features", label: "Feature Detail", icon: Search },
  { to: "/models", label: "Models", icon: Brain },
  { to: "/alerts", label: "Alerts", icon: Bell },
];

export default function Layout() {
  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="flex min-h-screen">
        <aside className="hidden w-64 border-r bg-muted/20 p-4 md:block">
          <div className="mb-8 flex items-center gap-2">
            <Activity className="h-6 w-6" />
            <div>
              <h1 className="font-bold leading-tight">ML Monitoring</h1>
              <p className="text-xs text-muted-foreground">Drift Dashboard</p>
            </div>
          </div>

          <nav className="space-y-2">
            {navItems.map((item) => {
              const Icon = item.icon;

              return (
                <NavLink
                  key={item.to}
                  to={item.to}
                  className={({ isActive }) =>
                    [
                      "flex items-center gap-2 rounded-md px-3 py-2 text-sm transition-colors",
                      isActive
                        ? "bg-primary text-primary-foreground"
                        : "text-muted-foreground hover:bg-muted hover:text-foreground",
                    ].join(" ")
                  }
                >
                  <Icon className="h-4 w-4" />
                  {item.label}
                </NavLink>
              );
            })}
          </nav>
        </aside>

        <main className="flex-1">
          <header className="border-b px-6 py-4">
            <h2 className="text-sm font-medium text-muted-foreground">
              ML Model Monitoring & Drift Detection Platform
            </h2>
          </header>

          <Outlet />
        </main>
      </div>
    </div>
  );
}