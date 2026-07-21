from passlib.hash import pbkdf2_sha256
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.dialects.postgresql import insert

from app.core.config import settings
from app.db.models.school import School
from app.db.models.user import User
from app.db.models.exam import Exam
from app.db.models.question_bank import Question
from app.db.base import Base


def get_engine():
    return create_async_engine(settings.DATABASE_URL, echo=False)


async def run_seeds():
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with async_session() as session:
        async with session.begin():
            await session.execute(
                insert(School)
                .values(name="Demo School", plan="basic")
                .on_conflict_do_nothing(index_elements=["name"])
            )

            await session.execute(
                insert(User)
                .values(
                    [
                        {
                            "school_id": 1,
                            "email": "admin@demo.edu",
                            "hashed_password": pbkdf2_sha256.hash("adminpass"),
                            "role": "admin",
                        },
                        {
                            "school_id": 1,
                            "email": "student@demo.edu",
                            "hashed_password": pbkdf2_sha256.hash("studentpass"),
                            "role": "student",
                        },
                    ]
                )
                .on_conflict_do_nothing(index_elements=["email"])
            )

            await session.execute(
                insert(Exam)
                .values(
                    school_id=1,
                    title="Introduction to Biology",
                    course="Biology",
                    duration_minutes=45,
                    is_active=True,
                )
                .on_conflict_do_nothing(index_elements=["title", "school_id"])
            )

            await session.execute(
                insert(Question)
                .values(
                    school_id=1,
                    course="Biology",
                    topic="Cell Structure",
                    question_data={
                        "question": "What is the powerhouse of the cell?",
                        "options": ["Nucleus", "Mitochondria", "Ribosome", "Golgi apparatus"],
                        "answer": "Mitochondria",
                    },
                )
                .on_conflict_do_nothing(index_elements=["school_id", "course", "topic"])
            )

    await engine.dispose()


if __name__ == "__main__":
    import asyncio

    asyncio.run(run_seeds())
