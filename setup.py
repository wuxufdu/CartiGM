from setuptools import find_packages, setup


setup(
    name="cartigsfm",
    version="0.4.0",
    description="Cartilage-domain gene-set foundation model utilities",
    long_description=open("README.md", encoding="utf-8").read() if __import__("pathlib").Path("README.md").exists() else "",
    long_description_content_type="text/markdown",
    author="CartiGSFM project",
    license="MIT",
    packages=find_packages(include=["cartigsfm", "cartigsfm.*"]),
    package_data={
        "cartigsfm": [
            "resources/dictionary_v1/*.json",
            "resources/rag_v1/*.json",
            "resources/p9_v1/config/*.json",
            "resources/p9_v1/docs/*.md",
            "resources/p9_v1/tsv/*.tsv",
        ]
    },
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=["numpy", "pandas"],
    entry_points={"console_scripts": ["cartigsfm=cartigsfm.cli:main"]},
)
