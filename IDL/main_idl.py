import sys
import re
import json
import subprocess
import threading
import urllib.error
import urllib.request
from pathlib import Path
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QPlainTextEdit, QWidget,
    QVBoxLayout, QHBoxLayout, QLabel, QSplitter,
    QTreeWidget, QTreeWidgetItem, QTabWidget, QInputDialog,
    QMessageBox, QPushButton, QFrame, QToolBar, QStatusBar
)
from PySide6.QtGui import (
    QColor, QFont, QTextCharFormat, QTextCursor,
    QSyntaxHighlighter, QPainter, QFontMetrics, QPalette
)
from PySide6.QtCore import Qt, QTimer, QRect, QSize, Signal, QProcess


# =========================
# FALLBACK LOCAL (nao usado pela API)
# =========================

def _predict_next_local(text):
    """
    Substitua pelo seu predict().
    Recebe o texto completo e retorna a sugestão como string.
    """
    last = text.split("\n")[-1].strip()
    suggestions = {
        "def soma(a, b):": "\n    return a + b",
        "for": " i in range(10):",
        "if": " True:",
        "class": " MinhaClasse:",
        "def": " funcao():",
        "return": " None",
        "import": " os",
        "print(": '"Hello, World!")',
    }
    for key, val in suggestions.items():
        if last == key or last.endswith(key):
            return val
    return ""


API_URL = "http://127.0.0.1:5000/generate"
API_TEMPERATURE = 1.0
API_MAX_TOKEN = 40
API_TIMEOUT = 20
MIN_TRIGGER_CHARS = 3
MAX_CONTEXT_LEVEL = 3
SCRIPT_DIR = Path(__file__).resolve().parent / "script"


def _request_completion(prompt, max_token=API_MAX_TOKEN):
    payload = {
        "prompt": prompt,
        "temperature": API_TEMPERATURE,
        "max_token": max_token,
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        API_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=API_TIMEOUT) as response:
        body = response.read().decode("utf-8")
        return json.loads(body).get("response", "")


def _completion_suffix(prompt, generated):
    if not generated:
        return ""

    if generated.startswith(prompt):
        return generated[len(prompt):]

    if generated.lower().startswith(prompt.lower()):
        return generated[len(prompt):]

    max_prefix = min(len(prompt), len(generated))
    for size in range(max_prefix, 0, -1):
        if generated[:size].lower() == prompt[-size:].lower():
            return generated[size:]

    return generated


def _strip_candidate_prefix(text, candidate):
    if not text or not candidate:
        return text

    candidate = candidate.strip()
    if not candidate:
        return text

    if text.lower().startswith(candidate.lower()):
        return text[len(candidate):]

    text_index = 0
    candidate_index = 0

    while text_index < len(text) and candidate_index < len(candidate):
        text_char = text[text_index]
        candidate_char = candidate[candidate_index]

        if candidate_char.isspace():
            while candidate_index < len(candidate) and candidate[candidate_index].isspace():
                candidate_index += 1
            while text_index < len(text) and text[text_index].isspace():
                text_index += 1
            continue

        if text_char.isspace():
            text_index += 1
            continue

        if text_char.lower() != candidate_char.lower():
            return text

        text_index += 1
        candidate_index += 1

    if candidate_index == len(candidate):
        return text[text_index:]

    return text


def _remove_prompt_echo(prompt, suggestion):
    if not suggestion:
        return ""

    current_line = prompt.split("\n")[-1].strip()
    candidates = [
        prompt.strip(),
        current_line,
    ]

    for candidate in sorted(candidates, key=len, reverse=True):
        cleaned = _strip_candidate_prefix(suggestion, candidate)
        if cleaned != suggestion:
            return cleaned.lstrip()

    return suggestion


def _current_token(text):
    match = re.search(r"[A-Za-z_][A-Za-z0-9_]*$", text)
    return match.group(0) if match else ""


def _current_indent(text):
    line = text.split("\n")[-1]
    match = re.match(r"^\s*", line)
    return match.group(0) if match else ""


def _find_enclosing_block(text):
    lines = text.splitlines()
    for index in range(len(lines) - 1, -1, -1):
        if re.match(r"^\s*(def|class)\s+\w+", lines[index]):
            return "\n".join(lines[index:])
    return ""


def _build_prediction_context(text, context_level):
    before_cursor = text.rstrip()
    if context_level <= 0:
        block = _find_enclosing_block(before_cursor)
        if block:
            return block[-900:]
        return "\n".join(before_cursor.splitlines()[-8:])

    if context_level == 1:
        block = _find_enclosing_block(before_cursor)
        if block:
            return block[-1600:]
        return "\n".join(before_cursor.splitlines()[-18:])

    if context_level == 2:
        return before_cursor[-2600:]

    return before_cursor[-5000:]


def _format_suggestion(suggestion, base_indent):
    if not suggestion:
        return ""

    text = suggestion.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r":\s+(?=(def|class|if|elif|else|for|while|try|except|finally|with|return|self\.|[A-Za-z_]))", ":\n    ", text)
    text = re.sub(r"\b(return\s+[^;\n]+?)\s+(?=(def|class|if|elif|else|for|while|try|except|finally|with|return)\b)", r"\1\n", text)
    text = re.sub(r"\s+(?=(def|class)\s+\w)", "\n", text)

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if not lines:
        return ""

    formatted = []
    indent = base_indent
    for index, line in enumerate(lines):
        if index == 0:
            formatted.append(line)
            if line.endswith(":"):
                indent = base_indent + "    "
            continue

        if re.match(r"^(elif|else|except|finally)\b", line):
            line_indent = base_indent
        else:
            line_indent = indent

        formatted.append(line_indent + line)
        if line.endswith(":"):
            indent = line_indent + "    "

    return "\n".join(formatted)


def predict_next(text, context_level=0):
    """
    Chama a API Flask local e retorna apenas o trecho novo para o ghost text.
    """
    prompt = text.rstrip()
    if not prompt or len(_current_token(prompt)) < MIN_TRIGGER_CHARS:
        return ""

    context = _build_prediction_context(prompt, context_level)
    max_token = API_MAX_TOKEN + (context_level * 30)

    try:
        generated = _request_completion(context, max_token=max_token)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return ""

    suffix = _completion_suffix(context, generated)
    suffix = _remove_prompt_echo(prompt, suffix)
    suffix = _format_suggestion(suffix, _current_indent(prompt))
    return suffix[:700]


# =========================
# CORES / TEMA
# =========================

THEME = {
    "bg0":      "#0d1117",
    "bg1":      "#161b22",
    "bg2":      "#1c2128",
    "bg3":      "#21262d",
    "bg4":      "#2d333b",
    "border":   "#30363d",
    "text":     "#e6edf3",
    "muted":    "#7d8590",
    "accent":   "#388bfd",
    "accent2":  "#58a6ff",
    "green":    "#3fb950",
    "orange":   "#d29922",
    "pink":     "#ff7b72",
    "purple":   "#d2a8ff",
    "teal":     "#79c0ff",
    "yellow":   "#e3b341",
    "ghost":    "#58a6ff",
}


# =========================
# SYNTAX HIGHLIGHTER
# =========================

KEYWORDS = {
    "False", "None", "True", "and", "as", "assert", "async", "await",
    "break", "class", "continue", "def", "del", "elif", "else", "except",
    "finally", "for", "from", "global", "if", "import", "in", "is",
    "lambda", "nonlocal", "not", "or", "pass", "raise", "return", "try",
    "while", "with", "yield"
}

BUILTINS = {
    "print", "len", "range", "int", "str", "float", "list", "dict",
    "tuple", "set", "bool", "type", "isinstance", "hasattr", "getattr",
    "setattr", "super", "property", "staticmethod", "classmethod",
    "enumerate", "zip", "map", "filter", "sorted", "reversed", "sum",
    "min", "max", "abs", "round", "open", "input", "format", "repr",
    "append", "extend", "insert", "remove", "pop", "update"
}


class PythonHighlighter(QSyntaxHighlighter):

    def __init__(self, document):
        super().__init__(document)
        self._build_formats()

    def _fmt(self, color, bold=False, italic=False):
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(color))
        if bold:
            fmt.setFontWeight(700)
        if italic:
            fmt.setFontItalic(True)
        return fmt

    def _build_formats(self):
        t = THEME
        self.fmt_keyword  = self._fmt(t["pink"], bold=True)
        self.fmt_builtin  = self._fmt(t["teal"])
        self.fmt_self     = self._fmt(t["orange"])
        self.fmt_string   = self._fmt(t["green"])
        self.fmt_comment  = self._fmt(t["muted"], italic=True)
        self.fmt_number   = self._fmt(t["yellow"])
        self.fmt_decorator= self._fmt(t["orange"])
        self.fmt_funcname = self._fmt(t["purple"])
        self.fmt_classname= self._fmt(t["yellow"], bold=True)
        self.fmt_ghost    = self._fmt(t["ghost"])
        self.fmt_ghost.setForeground(QColor(t["ghost"]))

        # Padrões como lista de (regex, formato)
        self.rules = []

        # Strings (triplas primeiro)
        for pat in [r'"""[\s\S]*?"""', r"'''[\s\S]*?'''",
                    r'"[^"\n]*"', r"'[^'\n]*'"]:
            self.rules.append((re.compile(pat), self.fmt_string))

        # f-strings
        self.rules.append((re.compile(r'f"[^"\n]*"'), self.fmt_string))
        self.rules.append((re.compile(r"f'[^'\n]*'"), self.fmt_string))

        # Comentários
        self.rules.append((re.compile(r'#[^\n]*'), self.fmt_comment))

        # Números
        self.rules.append((re.compile(r'\b\d+(\.\d+)?\b'), self.fmt_number))

        # Decoradores
        self.rules.append((re.compile(r'@\w+'), self.fmt_decorator))

        # Keywords
        kw_pat = r'\b(' + '|'.join(re.escape(k) for k in KEYWORDS) + r')\b'
        self.rules.append((re.compile(kw_pat), self.fmt_keyword))

        # Builtins
        bi_pat = r'\b(' + '|'.join(re.escape(b) for b in BUILTINS) + r')\b'
        self.rules.append((re.compile(bi_pat), self.fmt_builtin))

        # self / cls
        self.rules.append((re.compile(r'\b(self|cls)\b'), self.fmt_self))

        # Nome após def
        self.rules.append((re.compile(r'(?<=def )\w+'), self.fmt_funcname))

        # Nome após class
        self.rules.append((re.compile(r'(?<=class )\w+'), self.fmt_classname))

    def highlightBlock(self, text):
        for pattern, fmt in self.rules:
            for m in pattern.finditer(text):
                self.setFormat(m.start(), m.end() - m.start(), fmt)


# =========================
# NUMERAÇÃO DE LINHAS
# =========================

class LineNumberArea(QWidget):

    def __init__(self, editor):
        super().__init__(editor)
        self.editor = editor

    def sizeHint(self):
        return QSize(self.editor.line_number_area_width(), 0)

    def paintEvent(self, event):
        self.editor.line_number_area_paint_event(event)


# =========================
# EDITOR DE CÓDIGO
# =========================

class CodeEditor(QPlainTextEdit):

    ghost_accepted = Signal()
    prediction_ready = Signal(str, int)

    def __init__(self):
        super().__init__()
        self.ghost_text = ""
        self._setup_appearance()
        self.line_number_area = LineNumberArea(self)
        self.blockCountChanged.connect(self.update_line_number_area_width)
        self.updateRequest.connect(self.update_line_number_area)
        self.update_line_number_area_width(0)
        self._prediction_id = 0
        self._prediction_running = False
        self._context_level = 0
        self._suppress_until_len = 0
        self.prediction_ready.connect(self._on_prediction_ready)

        self.timer = QTimer()
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self._generate_prediction)
        self.textChanged.connect(lambda: self.timer.start(700))

    def _setup_appearance(self):
        t = THEME
        self.setStyleSheet(f"""
            QPlainTextEdit {{
                background-color: {t['bg0']};
                color: {t['text']};
                border: none;
                selection-background-color: {t['accent']};
            }}
        """)
        font = QFont("JetBrains Mono", 12)
        font.setStyleHint(QFont.StyleHint.Monospace)
        if not font.exactMatch():
            font = QFont("Fira Code", 12)
        if not font.exactMatch():
            font = QFont("Courier New", 12)
        self.setFont(font)
        self.setTabStopDistance(
            QFontMetrics(font).horizontalAdvance(' ') * 4
        )

    def line_number_area_width(self):
        digits = max(1, len(str(self.blockCount())))
        return 10 + self.fontMetrics().horizontalAdvance('9') * digits + 16

    def update_line_number_area_width(self, _=0):
        self.setViewportMargins(self.line_number_area_width(), 0, 0, 0)

    def update_line_number_area(self, rect, dy):
        if dy:
            self.line_number_area.scroll(0, dy)
        else:
            self.line_number_area.update(
                0, rect.y(),
                self.line_number_area.width(), rect.height()
            )
        if rect.contains(self.viewport().rect()):
            self.update_line_number_area_width(0)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        cr = self.contentsRect()
        self.line_number_area.setGeometry(
            QRect(cr.left(), cr.top(),
                  self.line_number_area_width(), cr.height())
        )

    def line_number_area_paint_event(self, event):
        t = THEME
        painter = QPainter(self.line_number_area)
        painter.fillRect(event.rect(), QColor(t["bg1"]))

        block = self.firstVisibleBlock()
        block_number = block.blockNumber()
        top = int(self.blockBoundingGeometry(block)
                  .translated(self.contentOffset()).top())
        bottom = top + int(self.blockBoundingRect(block).height())

        current_line = self.textCursor().blockNumber()

        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible() and bottom >= event.rect().top():
                number = str(block_number + 1)
                if block_number == current_line:
                    painter.setPen(QColor(t["text"]))
                else:
                    painter.setPen(QColor(t["muted"]))
                painter.drawText(
                    0, top,
                    self.line_number_area.width() - 8,
                    self.fontMetrics().height(),
                    Qt.AlignmentFlag.AlignRight, number
                )
            block = block.next()
            top = bottom
            bottom = top + int(self.blockBoundingRect(block).height())
            block_number += 1

    def keyPressEvent(self, event):
        key = event.key()

        # TAB: aceita sugestão ou insere 4 espaços
        if key == Qt.Key.Key_Tab:
            if self.ghost_text:
                cursor = self.textCursor()
                cursor.insertText(self.ghost_text)
                self.ghost_text = ""
                self._context_level = min(MAX_CONTEXT_LEVEL, self._context_level + 1)
                self.ghost_accepted.emit()
            else:
                cursor = self.textCursor()
                block_text = cursor.block().text()
                if block_text[:cursor.positionInBlock()].strip():
                    self._context_level = min(MAX_CONTEXT_LEVEL, self._context_level + 1)
                    self.timer.start(10)
                else:
                    cursor.insertText("    ")
            return

        # ESC: descarta sugestão
        if key == Qt.Key.Key_Escape:
            self.ghost_text = ""
            self._context_level = 0
            self._suppress_until_len = len(self.toPlainText()) + MIN_TRIGGER_CHARS
            self.ghost_accepted.emit()
            return

        # ENTER: auto-indentação
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            cursor = self.textCursor()
            block_text = cursor.block().text()
            indent_match = re.match(r'^(\s*)', block_text)
            indent = indent_match.group(1) if indent_match else ""
            if block_text.rstrip().endswith(":"):
                indent += "    "
            self.ghost_text = ""
            super().keyPressEvent(event)
            self.textCursor().insertText(indent)
            return

        super().keyPressEvent(event)
        if event.text().strip():
            self._context_level = 0

    def _generate_prediction(self):
        cursor = self.textCursor()
        text = self.toPlainText()[:cursor.position()]
        if len(text) < self._suppress_until_len:
            self.ghost_text = ""
            self.viewport().update()
            return

        if len(_current_token(text.rstrip())) < MIN_TRIGGER_CHARS:
            self.ghost_text = ""
            self.viewport().update()
            return

        self._prediction_id += 1
        prediction_id = self._prediction_id
        context_level = self._context_level
        self.ghost_text = ""
        self.viewport().update()

        if self._prediction_running:
            return

        self._prediction_running = True

        def worker():
            suggestion = predict_next(text, context_level=context_level)
            self.prediction_ready.emit(suggestion, prediction_id)

        threading.Thread(target=worker, daemon=True).start()

    def _on_prediction_ready(self, suggestion, prediction_id):
        self._prediction_running = False

        if prediction_id != self._prediction_id:
            self.timer.start(150)
            return

        self.ghost_text = suggestion
        self.viewport().update()

    def paintEvent(self, event):
        super().paintEvent(event)

        if not self.ghost_text:
            return

        cursor = self.textCursor()
        rect = self.cursorRect(cursor)

        painter = QPainter(self.viewport())
        painter.setPen(QColor(THEME["ghost"]))

        font = self.font()
        painter.setFont(font)
        painter.setOpacity(0.5)

        lines = self.ghost_text.split("\n")
        fm = self.fontMetrics()
        line_h = fm.height()

        x = rect.x()
        y = rect.top() + fm.ascent()

        for i, line in enumerate(lines):
            if i == 0:
                painter.drawText(x, y, line)
            else:
                y += line_h
                # indented ghost lines
                painter.drawText(
                    self.viewportMargins().left() + 4, y, line
                )

        painter.end()


# =========================
# PAINEL DE ARQUIVOS
# =========================

class FilePanel(QWidget):

    file_selected = Signal(str)
    file_created  = Signal(str)
    file_deleted  = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        t = THEME
        self.setStyleSheet(f"""
            QWidget {{ background: {t['bg1']}; }}
            QTreeWidget {{
                background: {t['bg1']};
                color: {t['text']};
                border: none;
                font-size: 12px;
            }}
            QTreeWidget::item {{ padding: 3px 6px; }}
            QTreeWidget::item:selected {{
                background: {t['bg4']};
                color: {t['accent2']};
            }}
            QTreeWidget::item:hover {{
                background: {t['bg3']};
            }}
            QPushButton {{
                background: transparent;
                color: {t['muted']};
                border: none;
                font-size: 16px;
                padding: 2px 6px;
            }}
            QPushButton:hover {{
                color: {t['text']};
                background: {t['bg3']};
                border-radius: 3px;
            }}
        """)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header
        header = QWidget()
        header.setFixedHeight(32)
        header.setStyleSheet(f"background: {THEME['bg1']}; border-bottom: 1px solid {THEME['border']};")
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(8, 0, 4, 0)

        lbl = QLabel("EXPLORADOR")
        lbl.setStyleSheet(f"color: {THEME['muted']}; font-size: 10px; font-weight: bold; letter-spacing: 1px;")
        h_layout.addWidget(lbl)
        h_layout.addStretch()

        btn_new = QPushButton("+")
        btn_new.setFixedSize(22, 22)
        btn_new.setToolTip("Novo arquivo")
        btn_new.clicked.connect(self._create_file)
        h_layout.addWidget(btn_new)

        btn_del = QPushButton("−")
        btn_del.setFixedSize(22, 22)
        btn_del.setToolTip("Excluir arquivo")
        btn_del.clicked.connect(self._delete_file)
        h_layout.addWidget(btn_del)

        layout.addWidget(header)

        # Tree
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.itemClicked.connect(
            lambda item: self.file_selected.emit(item.text(0))
        )
        layout.addWidget(self.tree)

    def populate(self, filenames, active=None):
        self.tree.clear()
        for name in filenames:
            item = QTreeWidgetItem([name])
            self.tree.addTopLevelItem(item)
            if name == active:
                self.tree.setCurrentItem(item)

    def _create_file(self):
        name, ok = QInputDialog.getText(
            self, "Novo arquivo", "Nome do arquivo (.py):"
        )
        if ok and name:
            if not name.endswith(".py"):
                name += ".py"
            self.file_created.emit(name)

    def _delete_file(self):
        item = self.tree.currentItem()
        if not item:
            return
        name = item.text(0)
        reply = QMessageBox.question(
            self, "Excluir", f"Excluir '{name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.file_deleted.emit(name)


class TerminalPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.process = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QWidget()
        header.setFixedHeight(34)
        header.setStyleSheet(f"background: {THEME['bg1']}; border-top: 1px solid {THEME['border']}; border-bottom: 1px solid {THEME['border']};")
        row = QHBoxLayout(header)
        row.setContentsMargins(8, 0, 8, 0)

        title = QLabel("TERMINAL")
        title.setStyleSheet(f"color: {THEME['muted']}; font-size: 10px; font-weight: bold;")
        row.addWidget(title)
        row.addStretch()

        self.btn_run = QPushButton("Rodar")
        self.btn_run.setToolTip("Executar o arquivo Python ativo")
        row.addWidget(self.btn_run)

        self.btn_open = QPushButton("Abrir terminal")
        self.btn_open.setToolTip("Abrir terminal na pasta script")
        row.addWidget(self.btn_open)

        layout.addWidget(header)

        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setStyleSheet(f"""
            QPlainTextEdit {{
                background-color: {THEME['bg0']};
                color: {THEME['text']};
                border: none;
                padding: 8px;
            }}
        """)
        font = QFont("JetBrains Mono", 10)
        font.setStyleHint(QFont.StyleHint.Monospace)
        self.output.setFont(font)
        layout.addWidget(self.output)

    def append(self, text):
        cursor = self.output.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(text)
        self.output.setTextCursor(cursor)

    def run_script(self, script_path):
        if self.process and self.process.state() != QProcess.ProcessState.NotRunning:
            self.process.kill()

        self.output.clear()
        self.append(f"$ python {script_path.name}\n\n")

        self.process = QProcess(self)
        self.process.setWorkingDirectory(str(script_path.parent))
        self.process.setProgram(sys.executable)
        self.process.setArguments([script_path.name])
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_output)
        self.process.finished.connect(self._finished)
        self.process.start()

    def _read_output(self):
        data = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self.append(data)

    def _finished(self, exit_code, _status):
        self.append(f"\n[processo finalizado com codigo {exit_code}]\n")

    def open_external(self, directory):
        subprocess.Popen(["cmd", "/K"], cwd=directory)


# =========================
# JANELA PRINCIPAL
# =========================

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Python AI Editor")
        self.resize(1200, 750)

        self.script_dir = SCRIPT_DIR
        self._ensure_script_dir()
        self.files = {
            "main.py":   'def main():\n    print("Olá, mundo!")\n\nif __name__ == "__main__":\n    main()\n',
            "modelo.py": '# Seu modelo aqui\n\ndef predict_next(text):\n    last = text.split("\\n")[-1].strip()\n    return ""\n',
            "utils.py":  'def soma(a, b):\n    return a + b\n\ndef fatorial(n):\n    if n <= 1:\n        return 1\n    return n * fatorial(n - 1)\n',
        }
        self.files = self._load_files_from_disk()
        self.active_file = "main.py"

        self._apply_theme()
        self._build_ui()
        self._load_file(self.active_file)

    def _ensure_script_dir(self):
        self.script_dir.mkdir(parents=True, exist_ok=True)

        main_path = self.script_dir / "main.py"
        example_path = self.script_dir / "exemplo.py"

        if not main_path.exists():
            main_path.write_text(
                'from exemplo import saudacao\n\n\n'
                'def main():\n'
                '    print(saudacao("IDL"))\n\n\n'
                'if __name__ == "__main__":\n'
                '    main()\n',
                encoding="utf-8",
            )

        if not example_path.exists():
            example_path.write_text(
                'def saudacao(nome):\n'
                '    return f"Ola, {nome}!"\n',
                encoding="utf-8",
            )

    def _load_files_from_disk(self):
        files = {}
        for path in sorted(self.script_dir.glob("*.py")):
            files[path.name] = path.read_text(encoding="utf-8")

        if "main.py" not in files:
            files["main.py"] = ""
        return files

    def _file_path(self, name):
        return self.script_dir / name

    def _apply_theme(self):
        t = THEME
        self.setStyleSheet(f"""
            QMainWindow {{ background: {t['bg0']}; }}
            QSplitter::handle {{ background: {t['border']}; width: 1px; }}
            QTabWidget::pane {{ border: none; background: {t['bg0']}; }}
            QTabBar::tab {{
                background: {t['bg1']};
                color: {t['muted']};
                padding: 6px 16px;
                border: none;
                border-right: 1px solid {t['border']};
                font-size: 12px;
            }}
            QTabBar::tab:selected {{
                background: {t['bg0']};
                color: {t['text']};
                border-bottom: 2px solid {t['accent']};
            }}
            QTabBar::tab:hover {{ color: {t['text']}; }}
            QStatusBar {{
                background: {t['accent']};
                color: white;
                font-size: 11px;
            }}
            QStatusBar QLabel {{ color: white; padding: 0 6px; }}
        """)

    def _build_ui(self):
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(1)

        # Painel de arquivos
        self.file_panel = FilePanel()
        self.file_panel.setMinimumWidth(160)
        self.file_panel.setMaximumWidth(280)
        self.file_panel.file_selected.connect(self._load_file)
        self.file_panel.file_created.connect(self._create_file)
        self.file_panel.file_deleted.connect(self._delete_file)
        splitter.addWidget(self.file_panel)

        # Área do editor com tabs
        right_splitter = QSplitter(Qt.Orientation.Vertical)
        right_splitter.setHandleWidth(1)

        self.tabs = QTabWidget()
        self.tabs.setTabsClosable(True)
        self.tabs.tabCloseRequested.connect(self._close_tab)
        self.tabs.currentChanged.connect(self._on_tab_changed)
        right_splitter.addWidget(self.tabs)

        self.terminal = TerminalPanel()
        self.terminal.btn_run.clicked.connect(self._run_active_file)
        self.terminal.btn_open.clicked.connect(lambda: self.terminal.open_external(self.script_dir))
        right_splitter.addWidget(self.terminal)
        right_splitter.setStretchFactor(0, 3)
        right_splitter.setStretchFactor(1, 1)
        right_splitter.setSizes([560, 190])

        splitter.addWidget(right_splitter)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([200, 900])

        self.setCentralWidget(splitter)

        # Status bar
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.lbl_lang   = QLabel("🐍 Python")
        self.lbl_lines  = QLabel("1 linha")
        self.lbl_pos    = QLabel("Ln 1, Col 1")
        self.lbl_hint   = QLabel("  TAB = aceitar sugestão  |  ESC = descartar")
        self.status.addWidget(self.lbl_lang)
        self.status.addWidget(self.lbl_lines)
        self.status.addPermanentWidget(self.lbl_hint)
        self.status.addPermanentWidget(self.lbl_pos)

        self._refresh_sidebar()

    # ─── Arquivos ───────────────────────────────

    def _refresh_sidebar(self):
        self.file_panel.populate(sorted(self.files.keys()), self.active_file)

    def _load_file(self, name):
        # Se já tem aba aberta, muda para ela
        for i in range(self.tabs.count()):
            if self.tabs.tabText(i) == name:
                self.tabs.setCurrentIndex(i)
                self.active_file = name
                self._refresh_sidebar()
                return

        # Cria nova aba
        editor = CodeEditor()
        PythonHighlighter(editor.document())
        editor.setPlainText(self.files.get(name, ""))
        editor.textChanged.connect(lambda: self._on_text_changed(name, editor))
        editor.cursorPositionChanged.connect(
            lambda: self._on_cursor_changed(editor)
        )
        editor.ghost_accepted.connect(lambda: editor.viewport().update())

        self.tabs.addTab(editor, name)
        self.tabs.setCurrentWidget(editor)
        self.active_file = name
        self._refresh_sidebar()
        self._update_status(editor)

    def _on_tab_changed(self, index):
        if index < 0:
            return
        name = self.tabs.tabText(index)
        self.active_file = name
        self._refresh_sidebar()
        editor = self.tabs.widget(index)
        if editor:
            self._update_status(editor)

    def _close_tab(self, index):
        if self.tabs.count() <= 1:
            return
        self.tabs.removeTab(index)
        self.active_file = self.tabs.tabText(self.tabs.currentIndex())
        self._refresh_sidebar()

    def _create_file(self, name):
        if name not in self.files:
            self.files[name] = f"# {name}\n"
            self._file_path(name).write_text(self.files[name], encoding="utf-8")
        self._load_file(name)

    def _delete_file(self, name):
        if len(self.files) <= 1:
            QMessageBox.warning(self, "Aviso", "Não é possível excluir o único arquivo.")
            return
        del self.files[name]
        path = self._file_path(name)
        if path.exists():
            path.unlink()
        # Fecha aba se estiver aberta
        for i in range(self.tabs.count()):
            if self.tabs.tabText(i) == name:
                self.tabs.removeTab(i)
                break
        if self.active_file == name:
            self.active_file = list(self.files.keys())[0]
            self._load_file(self.active_file)
        self._refresh_sidebar()

    def _save_active_file(self):
        index = self.tabs.currentIndex()
        if index < 0:
            return None

        name = self.tabs.tabText(index)
        editor = self.tabs.widget(index)
        if not editor:
            return None

        content = editor.toPlainText()
        self.files[name] = content
        path = self._file_path(name)
        path.write_text(content, encoding="utf-8")
        return path

    def _run_active_file(self):
        path = self._save_active_file()
        if path is None:
            return

        self.terminal.run_script(path)

    # ─── Status ─────────────────────────────────

    def _on_text_changed(self, name, editor):
        self.files[name] = editor.toPlainText()
        self._file_path(name).write_text(self.files[name], encoding="utf-8")
        lines = editor.blockCount()
        self.lbl_lines.setText(f"{lines} linha{'s' if lines != 1 else ''}")

    def _on_cursor_changed(self, editor):
        cursor = editor.textCursor()
        ln = cursor.blockNumber() + 1
        col = cursor.columnNumber() + 1
        self.lbl_pos.setText(f"Ln {ln}, Col {col}")

    def _update_status(self, editor):
        lines = editor.blockCount()
        self.lbl_lines.setText(f"{lines} linha{'s' if lines != 1 else ''}")
        cursor = editor.textCursor()
        ln = cursor.blockNumber() + 1
        col = cursor.columnNumber() + 1
        self.lbl_pos.setText(f"Ln {ln}, Col {col}")


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Paleta escura global
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window,      QColor(THEME["bg0"]))
    palette.setColor(QPalette.ColorRole.WindowText,  QColor(THEME["text"]))
    palette.setColor(QPalette.ColorRole.Base,        QColor(THEME["bg0"]))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(THEME["bg1"]))
    palette.setColor(QPalette.ColorRole.Text,        QColor(THEME["text"]))
    palette.setColor(QPalette.ColorRole.Button,      QColor(THEME["bg2"]))
    palette.setColor(QPalette.ColorRole.ButtonText,  QColor(THEME["text"]))
    palette.setColor(QPalette.ColorRole.Highlight,   QColor(THEME["accent"]))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    app.setPalette(palette)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())
