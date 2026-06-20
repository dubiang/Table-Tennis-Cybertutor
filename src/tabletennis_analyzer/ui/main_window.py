from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from tabletennis_analyzer.core.pipeline import analyze_user_coach_differences


class DifferenceAnalysisWorker(QObject):
    finished = Signal(str)
    failed = Signal(str)

    def __init__(self, user_video: Path, coach_video: Path, output_dir: Path, api_key: str, model: str) -> None:
        super().__init__()
        self.user_video = user_video
        self.coach_video = coach_video
        self.output_dir = output_dir
        self.api_key = api_key
        self.model = model

    def run(self) -> None:
        try:
            result = analyze_user_coach_differences(
                self.user_video,
                self.coach_video,
                self.output_dir,
                api_key=self.api_key,
                model=self.model,
            )
            lines = [
                "分析完成。",
                f"速度/加速度差函数: {result.difference_csv}",
                f"各关节差函数目录: {result.joint_difference_dir}",
                f"摘要数据: {result.difference_summary_json}",
            ]
            if result.llm_analysis_text:
                lines.extend(["", "GPT 分析:", result.llm_analysis_text])
            else:
                lines.append("未输入 API Key，已跳过 GPT 分析。")
            self.finished.emit("\n".join(lines))
        except Exception as exc:
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("乒乓球动作分析")
        self.resize(720, 520)

        self.user_video: Path | None = None
        self.coach_video: Path | None = None
        self.worker_thread: QThread | None = None
        self.worker: DifferenceAnalysisWorker | None = None

        self.user_button = QPushButton("上传自己的视频")
        self.user_button.clicked.connect(self.select_user_video)
        self.user_label = QLabel("未选择")

        self.coach_button = QPushButton("上传用于对照的视频")
        self.coach_button.clicked.connect(self.select_coach_video)
        self.coach_label = QLabel("未选择")

        self.api_input = QLineEdit()
        self.api_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_input.setPlaceholderText("键入 API Key")
        self.api_input.setFixedHeight(36)

        self.start_button = QPushButton("开始")
        self.start_button.clicked.connect(self.start_analysis)

        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setPlaceholderText("分析结果会显示在这里。")

        layout = QVBoxLayout()
        layout.setSpacing(12)
        layout.addWidget(self.user_button)
        layout.addWidget(self.user_label)
        layout.addWidget(self.coach_button)
        layout.addWidget(self.coach_label)
        layout.addWidget(self.api_input)
        layout.addWidget(self.start_button)
        layout.addWidget(self.output_text, stretch=1)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def select_user_video(self) -> None:
        path = self._select_mp4()
        if path:
            self.user_video = path
            self.user_label.setText(str(path))

    def select_coach_video(self) -> None:
        path = self._select_mp4()
        if path:
            self.coach_video = path
            self.coach_label.setText(str(path))

    def start_analysis(self) -> None:
        if not self.user_video or not self.coach_video:
            QMessageBox.warning(self, "缺少视频", "请先上传自己的视频和用于对照的视频。")
            return

        self.start_button.setEnabled(False)
        self.output_text.setPlainText("正在分析，请稍候...")
        self.worker_thread = QThread()
        self.worker = DifferenceAnalysisWorker(
            user_video=self.user_video,
            coach_video=self.coach_video,
            output_dir=Path.cwd() / "outputs" / "difference_analysis",
            api_key=self.api_input.text(),
            model="gpt-5.5",
        )
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.analysis_finished)
        self.worker.failed.connect(self.analysis_failed)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.failed.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.start()

    def analysis_finished(self, text: str) -> None:
        self.output_text.setPlainText(text)
        self.start_button.setEnabled(True)

    def analysis_failed(self, message: str) -> None:
        self.output_text.setPlainText("")
        self.start_button.setEnabled(True)
        QMessageBox.critical(self, "分析失败", message)

    def _select_mp4(self) -> Path | None:
        filename, _ = QFileDialog.getOpenFileName(self, "选择 MP4 视频", "", "MP4 Video (*.mp4)")
        return Path(filename) if filename else None


def run_app() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()

