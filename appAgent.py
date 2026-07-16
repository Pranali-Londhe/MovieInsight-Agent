from dotenv import load_dotenv
import streamlit as st

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_mistralai import ChatMistralAI

from pydantic import BaseModel
from typing import List, Optional


# Load Environment Variables
load_dotenv()


# ---------------- Movie Schema ---------------- #

class Movie(BaseModel):
    title: str
    release_year: Optional[int] = None
    genre: List[str]
    director: Optional[str] = None
    cast: List[str]
    rating: Optional[float] = None
    summary: str



# Parser
parser = PydanticOutputParser(
    pydantic_object=Movie
)


# LLM Model
model = ChatMistralAI(
    model="mistral-small-2506"
)



# Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are CineSage AI, an expert movie information extractor.

            Extract accurate movie details from the given paragraph.

            {format_instructions}
            """
        ),
        (
            "human",
            "{paragraph}"
        )
    ]
)



# ---------------- Streamlit Config ---------------- #

st.set_page_config(
    page_title="CineSage AI",
    page_icon="🎬",
    layout="wide"
)



# ---------------- Sidebar ---------------- #

with st.sidebar:

    st.image(
        "https://cdn-icons-png.flaticon.com/512/3658/3658896.png",
        width=100
    )

    st.title("🎬 CineSage AI")

    st.write(
        """
        An AI-powered movie information extraction system.

        Built using:
        
        🤖 Mistral AI  
        🔗 LangChain  
        📦 Pydantic Parser  
        🎨 Streamlit
        """
    )


    st.divider()

    st.info(
        "Paste any movie description and get structured movie insights."
    )




# ---------------- Main UI ---------------- #

st.title("🎬 CineSage AI")
st.subheader(
    "Intelligent Movie Information Extractor"
)


st.write(
    "Transform unstructured movie descriptions into structured insights using Generative AI."
)



paragraph = st.text_area(
    "📝 Enter Movie Description",
    height=250,
    placeholder=
    """
    Example:
    Interstellar is a 2014 science fiction movie directed by Christopher Nolan...
    """
)



col1, col2 = st.columns([1,1])


with col1:

    extract = st.button(
        "🚀 Extract Movie Details",
        use_container_width=True
    )


with col2:

    clear = st.button(
        "🧹 Clear",
        use_container_width=True
    )



if clear:
    st.session_state["paragraph"] = ""



# ---------------- Processing ---------------- #

if extract:

    if paragraph.strip():

        with st.spinner(
            "🎬 CineSage AI is analyzing the movie..."
        ):


            final_prompt = prompt.invoke(
                {
                    "paragraph": paragraph,
                    "format_instructions":
                    parser.get_format_instructions()
                }
            )


            response = model.invoke(
                final_prompt
            )


            try:

                movie = parser.parse(
                    response.content
                )


                st.success(
                    "✅ Movie Information Extracted Successfully"
                )


                st.divider()



                # Movie Header

                st.header(
                    f"🎥 {movie.title}"
                )


                # Basic Details

                c1,c2,c3 = st.columns(3)


                with c1:

                    st.metric(
                        "Release Year",
                        movie.release_year
                    )


                with c2:

                    st.metric(
                        "IMDb Rating",
                        movie.rating
                    )


                with c3:

                    st.metric(
                        "Director",
                        movie.director
                    )



                st.divider()



                # Genre

                st.subheader(
                    "🎭 Genre"
                )


                for genre in movie.genre:

                    st.success(
                        genre
                    )



                # Cast

                st.subheader(
                    "⭐ Cast"
                )


                cast_text = " | ".join(
                    movie.cast
                )

                st.info(
                    cast_text
                )



                # Summary

                st.subheader(
                    "📖 Story Summary"
                )


                st.write(
                    movie.summary
                )



                # JSON View

                with st.expander(
                    "📦 View Raw JSON"
                ):

                    st.json(
                        movie.model_dump()
                    )



            except Exception as e:

                st.error(
                    "Unable to parse response"
                )

                st.write(e)


    else:

        st.warning(
            "Please enter a movie paragraph first."
        )