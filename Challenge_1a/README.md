# Adobe Hackathon Round 1A: Document Structure Extraction

## Approach

This solution uses a multi-heuristic approach to detect document structure:

1. **Font Analysis**: Compares font sizes and weights to identify potential headings
2. **Position Analysis**: Uses bounding box information to detect formatting patterns  
3. **Pattern Matching**: Recognizes common heading patterns (numbered sections, keywords)
4. **Context Awareness**: Classifies heading levels based on document-wide patterns

## Key Featureswinget install Typora.Typora

- **Robust Heading Detection**: Combines multiple signals rather than relying on font size alone
- **Title Extraction**: Intelligently identifies document title from first page
- **Hierarchical Classification**: Assigns H1, H2, H3 levels based on patterns and context
- **Performance Optimized**: Processes 50-page PDFs in under 10 seconds
- **Language Agnostic**: Works with multilingual content including Japanese

## Libraries Used

- **PyMuPDF (fitz)**: Fast, accurate PDF text extraction with metadata
- **re**: Regular expressions for pattern matching
- **json**: Standard library for JSON output

## Build and Run

