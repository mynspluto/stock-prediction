"use client";

import { useEffect, useState } from "react";
import dynamic from "next/dynamic";

const DynamicChart = dynamic(() => import("../components/StockChart"), {
  ssr: false,
  loading: () => <div>Loading chart...</div>,
});

export default function Home() {
  return (
    <div>
      <DynamicChart />
    </div>
  );
}
