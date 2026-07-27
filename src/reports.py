import os
import csv
import io
from datetime import datetime

def generate_csv_log_export(csv_log_path):
    """
    Reads the crowd comparison CSV log file and returns its raw CSV string content.
    If the file does not exist, returns a formatted empty CSV header string.
    """
    if os.path.exists(csv_log_path):
        try:
            with open(csv_log_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            return f"timestamp,yolo_count,csrnet_count,which_model_selected,fill_rate,time_to_capacity\n# Error reading CSV: {e}\n"
    else:
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['timestamp', 'yolo_count', 'csrnet_count', 'which_model_selected', 'fill_rate', 'time_to_capacity'])
        return output.getvalue()

def generate_html_audit_report(metrics_cache, max_capacity, caution_at, alert_history=None):
    """
    Generates a dark-mode styled HTML Executive Safety Audit Report that can be printed
    directly or saved as PDF from the browser print dialog.
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    current_count = metrics_cache.get("current_count", 0)
    status = metrics_cache.get("status", "NORMAL")
    status_msg = metrics_cache.get("status_message", "System operating normally.")
    model_used = metrics_cache.get("model_used", "YOLO")
    fill_rate = metrics_cache.get("fill_rate", 0.0)
    time_to_cap = metrics_cache.get("time_to_capacity", 60.0)
    zone_grid = metrics_cache.get("zone_grid", [])
    risk_grid = metrics_cache.get("risk_per_zone", [])
    
    if alert_history is None:
        alert_history = metrics_cache.get("alert_history", [])
        
    capacity_pct = round((current_count / max(1, max_capacity)) * 100, 1)
    
    # Status color code
    status_color = "#10b981" if status == "NORMAL" else "#f59e0b" if status in ("WARNING", "EARLY", "URGENT") else "#ef4444"
    
    # Build Incident Log Table Rows
    log_rows_html = ""
    if alert_history:
        for item in reversed(alert_history[-20:]):  # Top 20 recent logs
            ts = item.get("time", now_str)
            tier = item.get("status", "INFO")
            msg = item.get("message", "")
            badge_color = "#10b981" if tier == "NORMAL" else "#f59e0b" if tier in ("WARNING", "EARLY", "URGENT") else "#ef4444"
            log_rows_html += f"""
            <tr>
                <td style="padding: 8px 12px; border-bottom: 1px solid #1e293b; color: #94a3b8; font-size: 11px;">{ts}</td>
                <td style="padding: 8px 12px; border-bottom: 1px solid #1e293b;"><span style="background: {badge_color}22; color: {badge_color}; border: 1px solid {badge_color}44; padding: 2px 8px; border-radius: 4px; font-size: 10px; font-weight: bold;">{tier}</span></td>
                <td style="padding: 8px 12px; border-bottom: 1px solid #1e293b; color: #e2e8f0; font-size: 12px;">{msg}</td>
            </tr>
            """
    else:
        log_rows_html = '<tr><td colspan="3" style="padding: 12px; text-align: center; color: #64748b;">No safety incidents or threshold alerts recorded during this session.</td></tr>'
        
    # Build 3x3 Grid HTML
    grid_cells_html = ""
    if len(zone_grid) == 3 and len(risk_grid) == 3:
        for r in range(3):
            for c in range(3):
                val = zone_grid[r][c]
                risk = risk_grid[r][c]
                risk_tier = risk.get('risk_tier', 'safe') if isinstance(risk, dict) else risk
                cell_color = "#ef4444" if risk_tier == 'danger' else "#f59e0b" if risk_tier == 'caution' else "#10b981"
                grid_cells_html += f"""
                <div style="background: #0f172a; border: 1px solid {cell_color}66; border-radius: 6px; padding: 12px; text-align: center;">
                    <div style="color: #94a3b8; font-size: 10px; text-transform: uppercase;">Zone ({r+1},{c+1})</div>
                    <div style="font-size: 20px; font-weight: bold; color: {cell_color}; margin: 4px 0;">{val:.1f}</div>
                    <div style="font-size: 9px; color: {cell_color}; font-weight: bold; text-transform: uppercase;">{risk_tier}</div>
                </div>
                """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>CrowdShield AI — Executive Safety Audit Report</title>
    <style>
        @media print {{
            body {{ background: #ffffff !important; color: #000000 !important; }}
            .card {{ border: 1px solid #cccccc !important; background: #ffffff !important; box-shadow: none !important; }}
            .no-print {{ display: none !important; }}
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #0b0f19;
            color: #f8fafc;
            margin: 0;
            padding: 24px;
        }}
        .report-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 2px solid #1e293b;
            padding-bottom: 16px;
            margin-bottom: 24px;
        }}
        .brand-title {{
            font-size: 24px;
            font-weight: bold;
            color: #38bdf8;
            margin: 0;
        }}
        .card {{
            background: #0f172a;
            border: 1px solid #1e293b;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 16px;
            margin-bottom: 20px;
        }}
        .stat-box {{
            background: #1e293b;
            padding: 16px;
            border-radius: 6px;
            text-align: center;
        }}
        .stat-val {{
            font-size: 24px;
            font-weight: bold;
            color: #38bdf8;
        }}
        .stat-lbl {{
            font-size: 11px;
            color: #94a3b8;
            text-transform: uppercase;
            margin-top: 4px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            text-align: left;
        }}
        th {{
            background: #1e293b;
            color: #94a3b8;
            padding: 10px 12px;
            font-size: 11px;
            text-transform: uppercase;
        }}
        .grid-3x3 {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
            margin-top: 12px;
        }}
        .btn-print {{
            background: #0284c7;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 6px;
            cursor: pointer;
            font-weight: bold;
            font-size: 13px;
        }}
    </style>
</head>
<body>

    <div class="report-header">
        <div>
            <h1 class="brand-title">🛡️ CrowdShield AI — Safety Audit Report</h1>
            <div style="color: #94a3b8; font-size: 12px; margin-top: 4px;">Executive Occupancy & Incident Log Summary</div>
        </div>
        <div style="text-align: right;">
            <button class="btn-print no-print" onclick="window.print()">🖨️ Print / Save as PDF</button>
            <div style="color: #64748b; font-size: 11px; margin-top: 6px;">Generated: {now_str}</div>
        </div>
    </div>

    <!-- STATS SUMMARY GRID -->
    <div class="stats-grid">
        <div class="stat-box">
            <div class="stat-val">{current_count:.0f}</div>
            <div class="stat-lbl">Occupancy Now</div>
        </div>
        <div class="stat-box">
            <div class="stat-val">{max_capacity}</div>
            <div class="stat-lbl">Max Capacity</div>
        </div>
        <div class="stat-box">
            <div class="stat-val" style="color: {status_color};">{capacity_pct}%</div>
            <div class="stat-lbl">Capacity Utilization</div>
        </div>
        <div class="stat-box">
            <div class="stat-val">{fill_rate:+.1f}/m</div>
            <div class="stat-lbl">Crowd Growth Rate</div>
        </div>
    </div>

    <!-- SYSTEM STATUS CARD -->
    <div class="card">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div style="font-size: 11px; color: #94a3b8; text-transform: uppercase;">Current Safety Status</div>
                <div style="font-size: 18px; font-weight: bold; color: {status_color}; margin-top: 2px;">{status}</div>
                <div style="font-size: 13px; color: #cbd5e1; margin-top: 4px;">{status_msg}</div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 11px; color: #94a3b8;">Active Model Engine</div>
                <div style="font-size: 16px; font-weight: bold; color: #38bdf8;">{model_used}</div>
                <div style="font-size: 11px; color: #94a3b8; margin-top: 4px;">Est. Time to Capacity: <strong style="color: #f8fafc;">{time_to_cap:.1f} mins</strong></div>
            </div>
        </div>
    </div>

    <!-- SPATIAL ZONE RISK BREAKDOWN -->
    <div class="card">
        <div style="font-size: 13px; font-weight: bold; color: #f8fafc; text-transform: uppercase;">Spatial Occupancy Grid Breakdown (3x3 Zones)</div>
        <div class="grid-3x3">
            {grid_cells_html}
        </div>
    </div>

    <!-- RECENT INCIDENT LOGS -->
    <div class="card">
        <div style="font-size: 13px; font-weight: bold; color: #f8fafc; text-transform: uppercase; margin-bottom: 12px;">Safety Incident Audit Trail</div>
        <table>
            <thead>
                <tr>
                    <th style="width: 160px;">Timestamp</th>
                    <th style="width: 100px;">Risk Tier</th>
                    <th>Audit Message</th>
                </tr>
            </thead>
            <tbody>
                {log_rows_html}
            </tbody>
        </table>
    </div>

</body>
</html>
"""
    return html
