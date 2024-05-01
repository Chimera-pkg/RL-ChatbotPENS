import asyncio

from prisma import Prisma
from prisma.models import Pertanyaan

async def simpan_pertanyaan() -> None:
    db = Prisma()
    db.connect()

    pertanyaan = db.pertanyaan.create(
        {
            'pertanyaan' :query
        }
    )

if __name__ == '__main__':
    asyncio.run(main())