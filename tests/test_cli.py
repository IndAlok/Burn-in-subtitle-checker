from burnin_subtitle_checker.cli import _metadata_matches, _ytdlp_auth_from_args, build_parser


class TestBuildParser:
    def test_transcribe_defaults(self):
        args = build_parser().parse_args(["transcribe", "video.mp4"])
        assert args.command == "transcribe"
        assert args.media == "video.mp4"
        assert args.language == "auto"
        assert args.model is None
        assert args.temperature == 0.0
        assert args.cookies_from_browser is None

    def test_ocr_defaults(self):
        args = build_parser().parse_args(["ocr", "video.mp4", "segments.json"])
        assert args.command == "ocr"
        assert args.crop_ratio == 0.15
        assert args.ocr_effort == "balanced"

    def test_compare_defaults(self):
        args = build_parser().parse_args(["compare", "audio.json", "ocr.json"])
        assert args.command == "compare"
        assert args.threshold == 0.80
        assert args.context_window_seconds == 2.0
        assert args.asset_mode == "link"
        assert args.output_csv == "outputs/report/results.csv"

    def test_report_defaults(self):
        args = build_parser().parse_args(["report", "outputs/report/results.json"])
        assert args.command == "report"
        assert args.results_json == "outputs/report/results.json"
        assert args.asset_mode == "link"
        assert args.output_html == "outputs/report/report.html"

    def test_run_defaults(self):
        args = build_parser().parse_args(["run", "video.mp4"])
        assert args.command == "run"
        assert args.language == "auto"
        assert args.model is None
        assert args.crop_ratio == 0.15
        assert args.threshold == 0.80
        assert args.samples_per_segment == 3
        assert args.ocr_effort == "balanced"
        assert not args.resume
        assert args.reuse_transcription is None
        assert args.reuse_ocr is None
        assert args.cookies_from_browser is None
        assert args.ytdlp_format == "bv*+ba/bestvideo+bestaudio/best/b"
        assert not args.recursive

    def test_run_with_options(self):
        args = build_parser().parse_args([
            "run", "video.mp4", "--language", "hi",
            "--model", "large-v3", "--temperature", "0.2", "--threshold", "0.80",
            "--crop-ratio", "0.20",
        ])
        assert args.language == "hi"
        assert args.model == "large-v3"
        assert args.temperature == 0.2
        assert args.threshold == 0.80
        assert args.crop_ratio == 0.20

    def test_run_ocr_language(self):
        args = build_parser().parse_args(["run", "video.mp4", "--ocr-language", "te", "--ocr-effort", "accurate"])
        assert args.ocr_language == "te"
        assert args.ocr_effort == "accurate"

    def test_tamil_and_telugu_language_choices(self):
        assert build_parser().parse_args(["transcribe", "video.mp4", "--language", "ta"]).language == "ta"
        assert build_parser().parse_args(["ocr", "video.mp4", "segments.json", "--language", "te"]).language == "te"

    def test_cookies_from_browser(self):
        args = build_parser().parse_args([
            "run", "https://youtube.com/watch?v=abc",
            "--cookies-from-browser", "chrome",
            "--cookies", "cookies.txt",
            "--user-agent", "Mozilla/5.0",
            "--referer", "https://www.youtube.com/",
            "--add-header", "Accept-Language: en-US",
            "--extractor-args", "youtube:player_client=mweb",
            "--proxy", "http://127.0.0.1:8080",
            "--source-address", "127.0.0.1",
            "--sleep-requests", "5",
            "--retries", "9",
            "--ytdlp-format", "best/b",
            "--format-sort", "res:720",
            "--js-runtimes", "deno",
            "--remote-components", "ejs:npm",
            "--prefer-yt-dlp-cli",
            "--list-formats",
        ])
        assert args.cookies_from_browser == "chrome"
        auth = _ytdlp_auth_from_args(args)
        assert auth.cookies == "cookies.txt"
        assert auth.user_agent == "Mozilla/5.0"
        assert auth.referer == "https://www.youtube.com/"
        assert auth.add_headers == ("Accept-Language: en-US",)
        assert auth.extractor_args == ("youtube:player_client=mweb",)
        assert auth.proxy == "http://127.0.0.1:8080"
        assert auth.source_address == "127.0.0.1"
        assert auth.sleep_requests == 5
        assert auth.retries == 9
        assert auth.format_selector == "best/b"
        assert auth.format_sort == ("res:720",)
        assert auth.js_runtimes == ("deno",)
        assert auth.remote_components == ("ejs:npm",)
        assert auth.prefer_cli is True
        assert args.list_formats

    def test_transcribe_cookies(self):
        args = build_parser().parse_args([
            "transcribe", "https://youtube.com/watch?v=abc",
            "--cookies-from-browser", "edge",
        ])
        assert args.cookies_from_browser == "edge"

    def test_resume_and_reuse_options(self):
        args = build_parser().parse_args([
            "run", "video.mp4", "--resume",
            "--reuse-transcription", "transcription.json",
            "--reuse-ocr", "ocr.json",
        ])
        assert args.resume
        assert args.reuse_transcription == "transcription.json"
        assert args.reuse_ocr == "ocr.json"

    def test_batch_and_preview_options(self):
        args = build_parser().parse_args([
            "run", "videos",
            "--recursive",
            "--include-all-files",
            "--continue-on-error",
            "--asset-mode", "preview",
        ])
        assert args.recursive
        assert args.include_all_files
        assert args.continue_on_error
        assert args.asset_mode == "preview"


class TestMetadataMatches:
    def test_matches_exact_and_float_values(self):
        payload = {"metadata": {"language": "hi", "temperature": 0.0}}
        assert _metadata_matches(payload, {"language": "hi", "temperature": 0.0})

    def test_rejects_mismatch(self):
        payload = {"metadata": {"language": "hi", "temperature": 0.0}}
        assert not _metadata_matches(payload, {"language": "kn"})
