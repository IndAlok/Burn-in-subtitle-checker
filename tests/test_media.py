import pytest

from burnin_subtitle_checker.media import (
    YtDlpAuth,
    _build_ytdlp_cli_args,
    _build_ytdlp_python_opts,
    _needs_ytdlp,
    _safe_filename,
    _validate_cookies_file,
    iter_media_files,
    is_url,
    safe_output_stem,
    stable_name,
)


class TestIsUrl:
    def test_http(self):
        assert is_url("http://example.com/video.mp4")

    def test_https(self):
        assert is_url("https://youtube.com/watch?v=abc")

    def test_local_path(self):
        assert not is_url("/path/to/video.mp4")

    def test_relative(self):
        assert not is_url("video.mp4")


class TestNeedsYtdlp:
    def test_youtube(self):
        assert _needs_ytdlp("https://www.youtube.com/watch?v=abc")

    def test_youtu_be(self):
        assert _needs_ytdlp("https://youtu.be/abc")

    def test_mobile_youtube(self):
        assert _needs_ytdlp("https://m.youtube.com/watch?v=abc")

    def test_direct_url(self):
        assert not _needs_ytdlp("https://example.com/video.mp4")

    def test_vimeo(self):
        assert not _needs_ytdlp("https://vimeo.com/12345")


class TestSafeFilename:
    def test_strips_special(self):
        assert "?" not in _safe_filename("what?is:this")

    def test_empty_fallback(self):
        assert _safe_filename("???") == "media"

    def test_max_len(self):
        long = "a" * 200
        assert len(_safe_filename(long, max_len=50)) == 50


class TestStableName:
    def test_deterministic(self):
        a = stable_name("https://example.com/video.mp4")
        b = stable_name("https://example.com/video.mp4")
        assert a == b

    def test_different_urls(self):
        a = stable_name("https://example.com/a.mp4")
        b = stable_name("https://example.com/b.mp4")
        assert a != b

    def test_suffix(self):
        name = stable_name("https://example.com/v.mp4", suffix=".wav")
        assert name.endswith(".wav")


class TestYtDlpAuth:
    def test_cookie_file_validation_accepts_netscape(self, tmp_path):
        cookie_file = tmp_path / "cookies.txt"
        cookie_file.write_text("# Netscape HTTP Cookie File\n.youtube.com\tTRUE\t/\tFALSE\t0\tVISITOR_INFO1_LIVE\tabc\n", encoding="utf-8")
        assert _validate_cookies_file(cookie_file) == cookie_file

    def test_cookie_file_validation_rejects_invalid_header(self, tmp_path):
        cookie_file = tmp_path / "cookies.txt"
        cookie_file.write_text("not a cookie header\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Mozilla/Netscape"):
            _validate_cookies_file(cookie_file)

    def test_cli_args_include_cookie_and_browser_options(self, tmp_path):
        cookie_file = tmp_path / "cookies.txt"
        cookie_file.write_text("# HTTP Cookie File\n", encoding="utf-8")
        auth = YtDlpAuth(
            cookies_from_browser="firefox:default-release",
            cookies=str(cookie_file),
            user_agent="Mozilla/5.0",
            referer="https://www.youtube.com/",
            add_headers=("Accept-Language: en-US,en;q=0.9",),
            extractor_args=("youtube:player_client=mweb",),
            proxy="http://127.0.0.1:8080",
            source_address="127.0.0.1",
            sleep_requests=5,
            retries=7,
            format_selector="best/b",
            format_sort=("res:720",),
            js_runtimes=("deno",),
            remote_components=("ejs:npm",),
            prefer_cli=True,
        )
        args = _build_ytdlp_cli_args(auth)
        assert args[args.index("--cookies-from-browser") + 1] == "firefox:default-release"
        assert args[args.index("--cookies") + 1] == str(cookie_file)
        assert args[args.index("--user-agent") + 1] == "Mozilla/5.0"
        assert args[args.index("--referer") + 1] == "https://www.youtube.com/"
        assert "Accept-Language: en-US,en;q=0.9" in args
        assert "youtube:player_client=mweb" in args
        assert args[args.index("--retries") + 1] == "7"
        assert args[args.index("--format-sort") + 1] == "res:720"
        assert args[args.index("--js-runtimes") + 1] == "deno"
        assert args[args.index("--remote-components") + 1] == "ejs:npm"

    def test_python_opts_parse_supported_auth(self, tmp_path):
        cookie_file = tmp_path / "cookies.txt"
        cookie_file.write_text("# Netscape HTTP Cookie File\n", encoding="utf-8")
        auth = YtDlpAuth(
            cookies_from_browser="chrome:Default",
            cookies=str(cookie_file),
            user_agent="Mozilla/5.0",
            add_headers=("Accept-Language: en-US",),
            extractor_args=("youtube:player_client=mweb;skip=webpage,configs",),
            format_selector="best/b",
            format_sort=("codec:h264",),
        )
        opts = _build_ytdlp_python_opts(auth)
        assert opts["format"] == "best/b"
        assert opts["format_sort"] == ["codec:h264"]
        assert opts["cookiesfrombrowser"] == ("chrome", "Default")
        assert opts["cookiefile"] == str(cookie_file)
        assert opts["user_agent"] == "Mozilla/5.0"
        assert opts["http_headers"]["Accept-Language"] == "en-US"
        assert opts["extractor_args"]["youtube"]["player_client"] == ["mweb"]
        assert opts["extractor_args"]["youtube"]["skip"] == ["webpage", "configs"]


class TestBatchMediaDiscovery:
    def test_iter_media_files_filters_known_media(self, tmp_path):
        (tmp_path / "a.mp4").write_text("", encoding="utf-8")
        (tmp_path / "b.mkv").write_text("", encoding="utf-8")
        (tmp_path / "notes.txt").write_text("", encoding="utf-8")

        files = [path.name for path in iter_media_files(tmp_path)]

        assert files == ["a.mp4", "b.mkv"]

    def test_iter_media_files_recursive_and_include_all(self, tmp_path):
        nested = tmp_path / "nested"
        nested.mkdir()
        (nested / "clip.webm").write_text("", encoding="utf-8")
        (tmp_path / "notes.txt").write_text("", encoding="utf-8")

        recursive = [path.name for path in iter_media_files(tmp_path, recursive=True)]
        all_files = [path.name for path in iter_media_files(tmp_path, include_all_files=True)]

        assert recursive == ["clip.webm"]
        assert all_files == ["notes.txt"]

    def test_safe_output_stem_uses_clean_local_stem(self):
        assert safe_output_stem("folder/My Clip?.mp4") == "My Clip"
