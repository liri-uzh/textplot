import pytest
import tempfile
import shutil
from pathlib import Path

from textplot.text import Text


class TestCorpusLoading:
    """Test suite for the different corpus loading strategies in the Text class."""

    @pytest.fixture(scope="function")
    def setup_teardown(self):
        """Set up and tear down test fixtures."""
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()

        # Create sample files for testing
        self.sample_files = [
            "file1.txt",
            "file2.txt",
            "subdir/file3.txt",
            "subdir/file4.md",
            "other.md",
        ]

        self.sample_texts = [
            "This is the first test file with some sample words.",
            "This is the second test file with different words.",
            "This third file contains nested directory content.",
            "This markdown file should be loadable with the right pattern.",
            "This MD file should not be loaded with the default pattern.",
        ]

        # Create the files with content
        for i, filepath in enumerate(self.sample_files):
            full_path = Path(self.test_dir) / filepath
            full_path.parent.mkdir(parents=True, exist_ok=True)
            with open(full_path, "w") as f:
                f.write(self.sample_texts[i])

        # Simple stopwords for testing
        self.stopwords = ["is", "the", "with", "this"]
        self.stopwords_file = Path(self.test_dir) / "stopwords.txt"
        with open(self.stopwords_file, "w") as f:
            f.write("\n".join(self.stopwords))

        yield (
            self.test_dir,
            self.sample_files,
            self.sample_texts,
            self.stopwords,
            self.stopwords_file,
        )

        # Clean up test fixtures
        shutil.rmtree(self.test_dir)

    def test_from_file_basic(self, setup_teardown):
        """Test loading a single file."""
        test_dir, sample_files, sample_texts, _, _ = setup_teardown
        file_path = Path(test_dir) / sample_files[0]
        text = Text.from_file(file_path)

        # Verify text content
        assert text.text == sample_texts[0]

        # Verify tokenization
        assert len(text.tokens) > 0
        assert len(text.terms) > 0

    def test_from_directory_all_files(self, setup_teardown):
        """Test loading all txt files from a directory."""
        test_dir, _, sample_texts, _, _ = setup_teardown
        text = Text.from_directory(test_dir)

        # Verify all .txt files were included
        for i, file_text in enumerate(sample_texts[:3]):  # First 3 are .txt
            assert file_text in text.text

        # Verify the .md file was not included with default pattern
        assert sample_texts[4] not in text.text

    def test_from_directory_custom_pattern(self, setup_teardown):
        """Test loading files with a custom pattern."""
        test_dir, _, sample_texts, _, _ = setup_teardown
        text = Text.from_directory(test_dir, file_pattern="*.md")

        # Verify only .md files were included
        assert sample_texts[3] in text.text
        assert sample_texts[4] in text.text

        # Verify .txt files were not included
        assert sample_texts[0] not in text.text

    def test_from_directory_non_recursive(self, setup_teardown):
        """Test loading files without recursion."""
        test_dir, _, sample_texts, _, _ = setup_teardown
        text = Text.from_directory(test_dir, recursive=False)

        # Verify only top-level .txt files were included
        assert sample_texts[0] in text.text
        assert sample_texts[1] in text.text

        # Verify nested directory files were not included
        assert sample_texts[2] not in text.text

    def test_from_texts_basic(self, setup_teardown):
        """Test loading from a list of strings."""
        _, _, sample_texts, _, _ = setup_teardown
        text = Text.from_texts(sample_texts[:2])

        # Verify text content includes both texts with separator
        expected = "\n\n".join(sample_texts[:2])
        assert text.text == expected

    def test_from_texts_custom_separator(self, setup_teardown):
        """Test loading from texts with a custom separator."""
        _, _, sample_texts, _, _ = setup_teardown
        separator = "\n---\n"
        text = Text.from_texts(sample_texts[:2], separator=separator)

        # Verify text content with custom separator
        expected = separator.join(sample_texts[:2])
        assert text.text == expected

    # def test_from_texts_with_kwargs(self, setup_teardown):
    #     """Test loading from texts with additional kwargs."""
    #     _, _, sample_texts, stopwords, stopwords_file = setup_teardown
    #     text = Text.from_texts(
    #         sample_texts[:2],
    #         stopwords=stopwords_file,
    #         use_spacy=True
    #     )

    #     # Verify additional args were passed through
    #     assert text.use_spacy
    #     for word in stopwords:
    #         assert word in text.stopwords
