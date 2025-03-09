import dynamic from "next/dynamic";
import axios from "axios";

// 서버 컴포넌트에서 데이터 가져오기
async function getStockData() {
  try {
    const response = await axios.get(
      `${process.env.NEXT_PUBLIC_API_URL}/predict/%5EIXIC`
    );
    return response.data;
  } catch (error) {
    console.error("Failed to fetch stock data:", error);
    return null;
  }
}

// 클라이언트 컴포넌트 동적 임포트
const DynamicChart = dynamic(() => import("../components/StockChart"), {
  ssr: false,
  loading: () => <div>Loading chart...</div>,
});

export default async function Home() {
  // 서버에서 데이터 가져오기
  const stockData = await getStockData();

  return (
    <div className="min-h-screen bg-gray-100 py-8">
      <DynamicChart initialData={stockData} />
    </div>
  );
}
