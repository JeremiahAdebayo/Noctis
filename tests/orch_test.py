import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Order:
    id: str
    user_id: str
    items: List[str]
    status: str = "pending"


class OrderEngine:
    def __init__(self):
        self.orders: Dict[str, Order] = {}
        self.user_active_order: Dict[str, str] = {}  # user_id -> order_id
        self._lock = asyncio.Lock()

    async def create_order(self, user_id: str, items: List[str]) -> Order:
        async with self._lock:
            # BUG AREA 1
            if user_id in self.user_active_order:
                existing_id = self.user_active_order[user_id]
                return self.orders[existing_id]

            order_id = f"order_{len(self.orders) + 1}"

            order = Order(
                id=order_id,
                user_id=user_id,
                items=items,
            )

            # simulate DB write delay
            await asyncio.sleep(0.05)

            self.orders[order_id] = order
            self.user_active_order[user_id] = order_id

            return order

    async def cancel_order(self, order_id: str) -> bool:
        async with self._lock:
            if order_id not in self.orders:
                return False

            order = self.orders[order_id]

            # BUG AREA 2
            if order.status == "completed":
                return False

            order.status = "cancelled"

            # simulate async DB update delay
            await asyncio.sleep(0.05)

            # BUG AREA 3
            if self.user_active_order.get(order.user_id) == order_id:
                del self.user_active_order[order.user_id]

            return True

    async def complete_order(self, order_id: str) -> bool:
        async with self._lock:
            if order_id not in self.orders:
                return False

            order = self.orders[order_id]

            if order.status == "cancelled":
                return False

            # simulate payment processing delay
            await asyncio.sleep(0.05)

            order.status = "completed"
            return True

    async def get_user_order(self, user_id: str) -> Optional[Order]:
        # BUG AREA 4 (no lock!)
        order_id = self.user_active_order.get(user_id)
        if not order_id:
            return None
        return self.orders.get(order_id)