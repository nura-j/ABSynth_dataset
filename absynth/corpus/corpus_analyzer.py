import numpy as np
from collections import Counter, defaultdict
import json

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional


class CorpusAnalyzer:
    """
    Advanced analysis tools for corpus research and development.
    """

    def __init__(self, corpus: Optional[List[Dict]] = None):
        """Initialize with optional corpus."""
        self.corpus = corpus

    def create_analysis_dataframe(self, corpus: Optional[List[Dict]] = None) -> pd.DataFrame:
        """
        Create a pandas DataFrame for advanced analysis.

        Args:
            corpus: Optional corpus to analyze

        Returns:
            DataFrame with sentence-level features
        """
        if corpus is None:
            corpus = self.corpus
        if not corpus:
            raise ValueError("No corpus provided")

        data = []
        for i, item in enumerate(corpus):
            metadata = item.get("metadata", {})
            semantic_roles = item.get("semantic_roles", {})

            row = {
                "sentence_id": i,
                "sentence": item["sentence"],
                "complexity": metadata.get("complexity", "unknown"),
                "frame": metadata.get("frame", "unknown"),
                "length": metadata.get("length", len(item["sentence"].split())),
                "avg_entropy": metadata.get("avg_entropy", 0),
                "num_roles": len(semantic_roles),
                "has_agent": any(info.get("role") == "Agent" for info in semantic_roles.values()),
                "has_patient": any(info.get("role") == "Patient" for info in semantic_roles.values()),
                "has_location": any(info.get("role") == "Location" for info in semantic_roles.values()),
            }

            # Add role-specific features
            for role_info in semantic_roles.values():
                role = role_info.get("role", "").lower()
                if role:
                    row[f"has_{role}"] = True

            data.append(row)

        return pd.DataFrame(data)

    def generate_correlation_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate correlation analysis between features."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlations = df[numeric_cols].corr()

        return {
            "correlation_matrix": correlations.to_dict(),
            "strongest_correlations": self._find_strongest_correlations(correlations),
            "feature_importance": self._calculate_feature_importance(df)
        }

    def _find_strongest_correlations(self, corr_matrix: pd.DataFrame) -> List[Dict[str, Any]]:
        """Find the strongest correlations in the matrix."""
        correlations = []

        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.5:  # Only strong correlations
                    correlations.append({
                        "feature1": corr_matrix.columns[i],
                        "feature2": corr_matrix.columns[j],
                        "correlation": corr_value,
                        "strength": "strong" if abs(corr_value) > 0.7 else "moderate"
                    })

        return sorted(correlations, key=lambda x: abs(x["correlation"]), reverse=True)

    def _calculate_feature_importance(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate feature importance for complexity prediction."""
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import LabelEncoder

            # Prepare features and target
            feature_cols = ["length", "avg_entropy", "num_roles", "has_agent", "has_patient", "has_location"]
            available_cols = [col for col in feature_cols if col in df.columns]

            if not available_cols or "complexity" not in df.columns:
                return {}

            X = df[available_cols].fillna(0)
            y = LabelEncoder().fit_transform(df["complexity"])

            # Train random forest
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X, y)

            return dict(zip(available_cols, rf.feature_importances_))

        except ImportError:
            return {"error": "scikit-learn not available for feature importance calculation"}

    def plot_corpus_overview(self, corpus: Optional[List[Dict]] = None, save_dir: Optional[str] = None):
        """
        Create comprehensive overview plots of the corpus.

        Args:
            corpus: Optional corpus to analyze
            save_dir: Optional directory to save plots
        """
        if corpus is None:
            corpus = self.corpus
        if not corpus:
            raise ValueError("No corpus provided")

        df = self.create_analysis_dataframe(corpus)

        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("Corpus Overview Analysis", fontsize=16)

        # 1. Complexity distribution
        complexity_counts = df["complexity"].value_counts()
        axes[0, 0].bar(complexity_counts.index, complexity_counts.values)
        axes[0, 0].set_title("Complexity Distribution")
        axes[0, 0].set_ylabel("Count")

        # 2. Frame distribution
        frame_counts = df["frame"].value_counts()
        axes[0, 1].bar(frame_counts.index, frame_counts.values)
        axes[0, 1].set_title("Semantic Frame Distribution")
        axes[0, 1].set_ylabel("Count")
        axes[0, 1].tick_params(axis='x', rotation=45)

        # 3. Sentence length distribution
        axes[0, 2].hist(df["length"], bins=20, alpha=0.7)
        axes[0, 2].set_title("Sentence Length Distribution")
        axes[0, 2].set_xlabel("Length (words)")
        axes[0, 2].set_ylabel("Frequency")

        # 4. Entropy distribution
        if "avg_entropy" in df.columns:
            axes[1, 0].hist(df["avg_entropy"], bins=20, alpha=0.7)
            axes[1, 0].set_title("Average Entropy Distribution")
            axes[1, 0].set_xlabel("Entropy")
            axes[1, 0].set_ylabel("Frequency")

        # 5. Semantic role coverage
        role_cols = [col for col in df.columns if col.startswith("has_")]
        if role_cols:
            role_counts = df[role_cols].sum().sort_values(ascending=False)
            axes[1, 1].bar(range(len(role_counts)), role_counts.values)
            axes[1, 1].set_title("Semantic Role Coverage")
            axes[1, 1].set_xlabel("Roles")
            axes[1, 1].set_ylabel("Count")
            axes[1, 1].set_xticks(range(len(role_counts)))
            axes[1, 1].set_xticklabels([col.replace("has_", "") for col in role_counts.index],
                                       rotation=45)

        # 6. Complexity vs Length relationship
        if len(df["complexity"].unique()) > 1:
            sns.boxplot(data=df, x="complexity", y="length", ax=axes[1, 2])
            axes[1, 2].set_title("Complexity vs Length")
            axes[1, 2].set_xlabel("Complexity")
            axes[1, 2].set_ylabel("Length")

        plt.tight_layout()

        if save_dir:
            plt.savefig(f"{save_dir}/corpus_overview.png", dpi=300, bbox_inches='tight')
            print(f"Overview plot saved to {save_dir}/corpus_overview.png")

        plt.show()

    def analyze_semantic_patterns(self, corpus: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Analyze semantic patterns and argument structures.

        Args:
            corpus: Optional corpus to analyze

        Returns:
            Dictionary with semantic pattern analysis
        """
        if corpus is None:
            corpus = self.corpus
        if not corpus:
            raise ValueError("No corpus provided")

        patterns = {
            "frame_role_combinations": Counter(),
            "argument_structures": Counter(),
            "role_transitions": Counter(),
            "complexity_frame_matrix": defaultdict(Counter)
        }

        for item in corpus:
            metadata = item.get("metadata", {})
            semantic_roles = item.get("semantic_roles", {})

            frame = metadata.get("frame", "unknown")
            complexity = metadata.get("complexity", "unknown")

            # Frame-role combinations
            roles = [info.get("role", "") for info in semantic_roles.values()]
            for role in roles:
                if role:
                    patterns["frame_role_combinations"][(frame, role)] += 1

            # Argument structures (number and types of arguments)
            arg_structure = tuple(sorted(roles))
            patterns["argument_structures"][arg_structure] += 1

            # Role transitions (order of roles in sentence)
            if len(roles) > 1:
                for i in range(len(roles) - 1):
                    if roles[i] and roles[i + 1]:
                        patterns["role_transitions"][(roles[i], roles[i + 1])] += 1

            # Complexity-frame relationships
            patterns["complexity_frame_matrix"][complexity][frame] += 1

        return {
            "most_common_frame_role_pairs": patterns["frame_role_combinations"].most_common(10),
            "most_common_argument_structures": patterns["argument_structures"].most_common(10),
            "most_common_role_transitions": patterns["role_transitions"].most_common(10),
            "complexity_frame_distribution": dict(patterns["complexity_frame_matrix"])
        }

    def export_analysis_report(self, corpus: Optional[List[Dict]] = None,
                               output_file: str = "corpus_analysis_report.json") -> Dict[str, Any]:
        """
        Generate and export comprehensive analysis report.

        Args:
            corpus: Optional corpus to analyze
            output_file: Output file path

        Returns:
            Complete analysis report
        """
        if corpus is None:
            corpus = self.corpus
        if not corpus:
            raise ValueError("No corpus provided")

        # Generate all analyses
        df = self.create_analysis_dataframe(corpus)
        correlations = self.generate_correlation_analysis(df)
        semantic_patterns = self.analyze_semantic_patterns(corpus)

        # Basic statistics
        basic_stats = {
            "total_sentences": len(corpus),
            "avg_sentence_length": df["length"].mean(),
            "complexity_distribution": df["complexity"].value_counts().to_dict(),
            "frame_distribution": df["frame"].value_counts().to_dict(),
            "semantic_role_coverage": {
                col.replace("has_", ""): df[col].sum()
                for col in df.columns if col.startswith("has_")
            }
        }

        # Compile full report
        report = {
            "analysis_metadata": {
                "corpus_size": len(corpus),
                "analysis_timestamp": pd.Timestamp.now().isoformat(),
                "features_analyzed": list(df.columns)
            },
            "basic_statistics": basic_stats,
            "correlation_analysis": correlations,
            "semantic_patterns": semantic_patterns,
            "quality_metrics": {
                "semantic_diversity": len(df["frame"].unique()),
                "complexity_balance": self._calculate_balance_score(df["complexity"].value_counts()),
                "role_coverage_score": len(
                    [col for col in df.columns if col.startswith("has_") and df[col].sum() > 0]) / 9
            }
        }

        # Save report
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        print(f"Analysis report exported to {output_file}")
        return report

    def _calculate_balance_score(self, value_counts: pd.Series) -> float:
        """Calculate how balanced a distribution is (1.0 = perfectly balanced)."""
        if len(value_counts) == 0:
            return 0.0

        total = value_counts.sum()
        expected = total / len(value_counts)

        # Calculate deviation from expected uniform distribution
        deviations = [abs(count - expected) / expected for count in value_counts]
        avg_deviation = sum(deviations) / len(deviations)

        # Convert to balance score (1.0 = perfect balance, 0.0 = maximum imbalance)
        return max(0.0, 1.0 - avg_deviation)

    def find_outliers(self, corpus: Optional[List[Dict]] = None, feature: str = "length") -> List[Dict[str, Any]]:
        """
        Find statistical outliers in the corpus.

        Args:
            corpus: Optional corpus to analyze
            feature: Feature to analyze for outliers

        Returns:
            List of outlier sentences with details
        """
        if corpus is None:
            corpus = self.corpus
        if not corpus:
            raise ValueError("No corpus provided")

        df = self.create_analysis_dataframe(corpus)

        if feature not in df.columns:
            raise ValueError(f"Feature '{feature}' not found in corpus")

        # Calculate outlier thresholds using IQR method
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Find outliers
        outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]

        outlier_details = []
        for _, row in outliers.iterrows():
            sentence_idx = int(row["sentence_id"])
            outlier_details.append({
                "sentence_id": sentence_idx,
                "sentence": row["sentence"],
                "feature_value": row[feature],
                "outlier_type": "low" if row[feature] < lower_bound else "high",
                "deviation_from_median": abs(row[feature] - df[feature].median()),
                "complexity": row.get("complexity", "unknown"),
                "frame": row.get("frame", "unknown")
            })

        return sorted(outlier_details, key=lambda x: x["deviation_from_median"], reverse=True)

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from typing import List, Dict, Any, Optional
#
#
# class CorpusAnalyzer:
#     """
#     Advanced analysis tools for corpus research and development.
#     """
#
#     def __init__(self, corpus: Optional[List[Dict]] = None):
#         """Initialize with optional corpus."""
#         self.corpus = corpus
#
#     def create_analysis_dataframe(self, corpus: Optional[List[Dict]] = None) -> pd.DataFrame:
#         """
#         Create a pandas DataFrame for advanced analysis.
#
#         Args:
#             corpus: Optional corpus to analyze
#
#         Returns:
#             DataFrame with sentence-level features
#         """
#         if corpus is None:
#             corpus = self.corpus
#         if not corpus:
#             raise ValueError("No corpus provided")
#
#         data = []
#         for i, item in enumerate(corpus):
#             metadata = item.get("metadata", {})
#             semantic_roles = item.get("semantic_roles", {})
#
#             row = {
#                 "sentence_id": i,
#                 "sentence": item["sentence"],
#                 "complexity": metadata.get("complexity", "unknown"),
#                 "frame": metadata.get("frame", "unknown"),
#                 "length": metadata.get("length", len(item["sentence"].split())),
#                 "avg_entropy": metadata.get("avg_entropy", 0),
#                 "num_roles": len(semantic_roles),
#                 "has_agent": any(info.get("role") == "Agent" for info in semantic_roles.values()),
#                 "has_patient": any(info.get("role") == "Patient" for info in semantic_roles.values()),
#                 "has_location": any(info.get("role") == "Location" for info in semantic_roles.values()),
#             }
#
#             # Add role-specific features
#             for role_info in semantic_roles.values():
#                 role = role_info.get("role", "").lower()
#                 if role:
#                     row[f"has_{role}"] = True
#
#             data.append(row)
#
#         return pd.DataFrame(data)
#
#     def generate_correlation_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
#         """Generate correlation analysis between features."""
#         numeric_cols = df.select_dtypes(include=[np.number]).columns
#         correlations = df[numeric_cols].corr()
#
#         return {
#             "correlation_matrix": correlations.to_dict(),
#             "strongest_correlations": self._find_strongest_correlations(correlations),
#             "feature_importance": self._calculate_feature_importance(df)
#         }
#
#     def _find_strongest_correlations(self, corr_matrix: pd.DataFrame) -> List[Dict[str, Any]]:
#         """Find the strongest correlations in the matrix."""
#         correlations = []
#
#         for i in range(len(corr_matrix.columns)):
#             for j in range(i + 1, len(corr_matrix.columns)):
#                 corr_value = corr_matrix.iloc[i, j]
#                 if abs(corr_value) > 0.5:  # Only strong correlations
#                     correlations.append({
#                         "feature1": corr_matrix.columns[i],
#                         "feature2": corr_matrix.columns[j],
#                         "correlation": corr_value,
#                         "strength": "strong" if abs(corr_value) > 0.7 else "moderate"
#                     })
#
#         return sorted(correlations, key=lambda x: abs(x["correlation"]), reverse=True)
#
#     def _calculate_feature_importance(self, df: pd.DataFrame) -> Dict[str, float]:
#         """Calculate feature importance for complexity prediction."""
#         try:
#             from sklearn.ensemble import RandomForestClassifier
#             from sklearn.preprocessing import LabelEncoder
#
#             # Prepare features and target
#             feature_cols = ["length", "avg_entropy", "num_roles", "has_agent", "has_patient", "has_location"]
#             available_cols = [col for col in feature_cols if col in df.columns]
#
#             if not available_cols or "complexity" not in df.columns:
#                 return {}
#
#             X = df[available_cols].fillna(0)
#             y = LabelEncoder().fit_transform(df["complexity"])
#
#             # Train random forest
#             rf = RandomForestClassifier(n_estimators=100, random_state=42)
#             rf.fit(X, y)
#
#             return dict(zip(available_cols, rf.feature_importances_))
#
#         except ImportError:
#             return {"error": "scikit-learn not available for feature importance calculation"}
#
#     def plot_corpus_overview(self, corpus: Optional[List[Dict]] = None, save_dir: Optional[str] = None):
#         """
#         Create comprehensive overview plots of the corpus.
#
#         Args:
#             corpus: Optional corpus to analyze
#             save_dir: Optional directory to save plots
#         """
#         if corpus is None:
#             corpus = self.corpus
#         if not corpus:
#             raise ValueError("No corpus provided")
#
#         df = self.create_analysis_dataframe(corpus)
#
#         # Create subplots
#         fig, axes = plt.subplots(2, 3, figsize=(18, 12))
#         fig.suptitle("Corpus Overview Analysis", fontsize=16)
#
#         # 1. Complexity distribution
#         complexity_counts = df["complexity"].value_counts()
#         axes[0, 0].bar(complexity_counts.index, complexity_counts.values)
#         axes[0, 0].set_title("Complexity Distribution")
#         axes[0, 0].set_ylabel("Count")
#
#         # 2. Frame distribution
#         frame_counts = df["frame"].value_counts()
#         axes[0, 1].bar(frame_counts.index, frame_counts.values)
#         axes[0, 1].set_title("Semantic Frame Distribution")
#         axes[0, 1].set_ylabel("Count")
#         axes[0, 1].tick_params(axis='x', rotation=45)
#
#         # 3. Sentence length distribution
#         axes[0, 2].hist(df["length"], bins=20, alpha=0.7)
#         axes[0, 2].set_title("Sentence Length Distribution")
#         axes[0, 2].set_xlabel("Length (words)")
#         axes[0, 2].set_ylabel("Frequency")
#
#         # 4. Entropy distribution
#         if "avg_entropy" in df.columns:
#             axes[1, 0].hist(df["avg_entropy"], bins=20, alpha=0.7)
#             axes[1, 0].set_title("Average Entropy Distribution")
#             axes[1, 0].set_xlabel("Entropy")
#             axes[1, 0].set_ylabel("Frequency")
#
#         # 5. Semantic role coverage
#         role_cols = [col for col in df.columns if col.startswith("has_")]
#         if role_cols:
#             role_counts = df[role_cols].sum().sort_values(ascending=False)
#             axes[1, 1].bar(range(len(role_counts)), role_counts.values)
#             axes[1, 1].set_title("Semantic Role Coverage")
#             axes[1, 1].set_xlabel("Roles")
#             axes[1, 1].set_ylabel("Count")
#             axes[1, 1].set_xticks(range(len(role_counts)))
#             axes[1, 1].set_xticklabels([col.replace("has_", "") for col in role_counts.index],
#                                        rotation=45)
#
#         # 6. Complexity vs Length relationship
#         if len(df["complexity"].unique()) > 1:
#             sns.boxplot(data=df, x="complexity", y="length", ax=axes[1, 2])
#             axes[1, 2].set_title("Complexity vs Length")
#             axes[1, 2].set_xlabel("Complexity")
#             axes[1, 2].set_ylabel("Length")
#
#         plt.tight_layout()
#
#         if save_dir:
#             plt.savefig(f"{save_dir}/corpus_overview.png", dpi=300, bbox_inches='tight')
#             print(f"Overview plot saved to {save_dir}/corpus_overview.png")
#
#         plt.show()
#
#     def analyze_semantic_patterns(self, corpus: Optional[List[Dict]] = None) -> Dict[str, Any]:
#         """
#         Analyze semantic patterns and argument structures.
#
#         Args:
#             corpus: Optional corpus to analyze
#
#         Returns:
#             Dictionary with semantic pattern analysis
#         """
#         if corpus is None:
#             corpus = self.corpus
#         if not corpus:
#             raise ValueError("No corpus provided")
#
#         patterns = {
#             "frame_role_combinations": Counter(),
#             "argument_structures": Counter(),
#             "role_transitions": Counter(),
#             "complexity_frame_matrix": defaultdict(Counter)
#         }
#
#         for item in corpus:
#             metadata = item.get("metadata", {})
#             semantic_roles = item.get("semantic_roles", {})
#
#             frame = metadata.get("frame", "unknown")
#             complexity = metadata.get("complexity", "unknown")
#
#             # Frame-role combinations
#             roles = [info.get("role", "") for info in semantic_roles.values()]
#             for role in roles:
#                 if role:
#                     patterns["frame_role_combinations"][(frame, role)] += 1
#
#             # Argument structures (number and types of arguments)
#             arg_structure = tuple(sorted(roles))
#             patterns["argument_structures"][arg_structure] += 1
#
#             # Role transitions (order of roles in sentence)
#             if len(roles) > 1:
#                 for i in range(len(roles) - 1):
#                     if roles[i] and roles[i + 1]:
#                         patterns["role_transitions"][(roles[i], roles[i + 1])] += 1
#
#             # Complexity-frame relationships
#             patterns["complexity_frame_matrix"][complexity][frame] += 1
#
#         return {
#             "most_common_frame_role_pairs": patterns["frame_role_combinations"].most_common(10),
#             "most_common_argument_structures": patterns["argument_structures"].most_common(10),
#             "most_common_role_transitions": patterns["role_transitions"].most_common(10),
#             "complexity_frame_distribution": dict(patterns["complexity_frame_matrix"])
#         }
#
#     def export_analysis_report(self, corpus: Optional[List[Dict]] = None,
#                                output_file: str = "corpus_analysis_report.json") -> Dict[str, Any]:
#         """
#         Generate and export comprehensive analysis report.
#
#         Args:
#             corpus: Optional corpus to analyze
#             output_file: Output file path
#
#         Returns:
#             Complete analysis report
#         """
#         if corpus is None:
#             corpus = self.corpus
#         if not corpus:
#             raise ValueError("No corpus provided")
#
#         # Generate all analyses
#         df = self.create_analysis_dataframe(corpus)
#         correlations = self.generate_correlation_analysis(df)
#         semantic_patterns = self.analyze_semantic_patterns(corpus)
#
#         # Basic statistics
#         basic_stats = {
#             "total_sentences": len(corpus),
#             "avg_sentence_length": df["length"].mean(),
#             "complexity_distribution": df["complexity"].value_counts().to_dict(),
#             "frame_distribution": df["frame"].value_counts().to_dict(),
#             "semantic_role_coverage": {
#                 col.replace("has_", ""): df[col].sum()
#                 for col in df.columns if col.startswith("has_")
#             }
#         }
#
#         # Compile full report
#         report = {
#             "analysis_metadata": {
#                 "corpus_size": len(corpus),
#                 "analysis_timestamp": pd.Timestamp.now().isoformat(),
#                 "features_analyzed": list(df.columns)
#             },
#             "basic_statistics": basic_stats,
#             "correlation_analysis": correlations,
#             "semantic_patterns": semantic_patterns,
#             "quality_metrics": {
#                 "semantic_diversity": len(df["frame"].unique()),
#                 "complexity_balance": self._calculate_balance_score(df["complexity"].value_counts()),
#                 "role_coverage_score": len(
#                     [col for col in df.columns if col.startswith("has_") and df[col].sum() > 0]) / 9
#             }
#         }
#
#         # Save report
#         with open(output_file, 'w', encoding='utf-8') as f:
#             json.dump(report, f, indent=2, ensure_ascii=False, default=str)
#
#         print(f"Analysis report exported to {output_file}")
#         return report
#
#     def _calculate_balance_score(self, value_counts: pd.Series) -> float:
#         """Calculate how balanced a distribution is (1.0 = perfectly balanced)."""
#         if len(value_counts) == 0:
#             return 0.0
#
#         total = value_counts.sum()
#         expected = total / len(value_counts)
#
#         # Calculate deviation from expected uniform distribution
#         deviations = [abs(count - expected) / expected for count in value_counts]
#         avg_deviation = sum(deviations) / len(deviations)
#
#         # Convert to balance score (1.0 = perfect balance, 0.0 = maximum imbalance)
#         return max(0.0, 1.0 - avg_deviation)
#
#     def find_outliers(self, corpus: Optional[List[Dict]] = None, feature: str = "length") -> List[Dict[str, Any]]:
#         """
#         Find statistical outliers in the corpus.
#
#         Args:
#             corpus: Optional corpus to analyze
#             feature: Feature to analyze for outliers
#
#         Returns:
#             List of outlier sentences with details
#         """
#         if corpus is None:
#             corpus = self.corpus
#         if not corpus:
#             raise ValueError("No corpus provided")
#
#         df = self.create_analysis_dataframe(corpus)
#
#         if feature not in df.columns:
#             raise ValueError(f"Feature '{feature}' not found in corpus")
#
#         # Calculate outlier thresholds using IQR method
#         Q1 = df[feature].quantile(0.25)
#         Q3 = df[feature].quantile(0.75)
#         IQR = Q3 - Q1
#         lower_bound = Q1 - 1.5 * IQR
#         upper_bound = Q3 + 1.5 * IQR
#
#         # Find outliers
#         outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]
#
#         outlier_details = []
#         for _, row in outliers.iterrows():
#             sentence_idx = int(row["sentence_id"])
#             outlier_details.append({
#                 "sentence_id": sentence_idx,
#                 "sentence": row["sentence"],
#                 "feature_value": row[feature],
#                 "outlier_type": "low" if row[feature] < lower_bound else "high",
#                 "deviation_from_median": abs(row[feature] - df[feature].median()),
#                 "complexity": row.get("complexity", "unknown"),
#                 "frame": row.get("frame", "unknown")
#             })
#
#         return sorted(outlier_details, key=lambda x: x["deviation_from_median"], reverse=True)