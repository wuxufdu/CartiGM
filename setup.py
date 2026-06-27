from setuptools import find_packages, setup


setup(
    name="cartigsfm",
    version="0.6.1",
    description="Cartilage-domain gene-set foundation model utilities",
    long_description=open("README.md", encoding="utf-8").read() if __import__("pathlib").Path("README.md").exists() else "",
    long_description_content_type="text/markdown",
    author="CartiGSFM project",
    license="MIT",
    packages=find_packages(include=["cartigsfm", "cartigsfm.*", "cartigsfm_web", "cartigsfm_web.*"]),
    package_data={
        "cartigsfm": [
            "resources/dictionary_v1/*.json",
            "resources/rag_v1/*.json",
            "resources/p9_v1/config/*.json",
            "resources/p9_v1/docs/*.md",
            "resources/p9_v1/tsv/*.tsv",
            "resources/cs_classifier_v1/*.pt",
            "resources/cs_classifier_v1/*.tsv",
        ],
        "cartigsfm_web": [
            "static/*.html",
            "static/*.css",
            "static/*.js",
        ],
    },
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=["numpy", "pandas"],
    extras_require={"web": ["fastapi", "uvicorn"]},
    entry_points={
        "console_scripts": [
            "cartigsfm=cartigsfm.cli:main",
            "cartigsfm-web=cartigsfm_web.server:main",
        ]
    },
)
